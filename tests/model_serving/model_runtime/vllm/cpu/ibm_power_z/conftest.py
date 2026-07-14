from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.vllm.cpu.ibm_power_z.constant import (
    IBM_POWER_Z_ENV_VARIABLES,
    IBM_POWER_Z_PREDICT_RESOURCES,
)
from tests.model_serving.model_runtime.vllm.utils import (
    add_image_pull_secrets_if_configured,
    dedupe_vllm_cli_args,
    skip_if_not_deployment_mode,
    validate_supported_quantization_schema,
)
from utilities.constants import AcceleratorType, KServeDeploymentType, Timeout
from utilities.inference_utils import create_isvc

LOGGER = structlog.get_logger(name=__name__)

SUPPORTED_IBM_POWER_Z_ACCELERATORS: set[str] = {
    AcceleratorType.CPU_POWER,
    AcceleratorType.CPU_Z,
}


@pytest.fixture(scope="session")
def skip_if_no_supported_ibm_power_z_accelerator_type(
    supported_accelerator_type: str | None,
) -> None:
    """Skip test unless the cluster provides a supported IBM Power or Z CPU accelerator."""
    if (
        not supported_accelerator_type
        or supported_accelerator_type.lower()
        not in SUPPORTED_IBM_POWER_Z_ACCELERATORS
    ):
        pytest.skip(
            f"Test requires a supported vLLM IBM Power or Z CPU accelerator. "
            f"Found: '{supported_accelerator_type or 'None'}'. "
            f"Expected one of: {SUPPORTED_IBM_POWER_Z_ACCELERATORS}."
        )


@pytest.fixture(scope="class")
def ibm_power_z_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    skip_if_no_supported_ibm_power_z_accelerator_type: None,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime, None, None]:
    """ServingRuntime for vLLM CPU Power or Z."""

    LOGGER.info(
        f"Creating vLLM CPU ServingRuntime for {supported_accelerator_type}"
    )

    serving_runtime = ServingRuntime(
        client=admin_client,
        name="vllm-runtime",
        namespace=model_namespace.name,
        annotations={
            "opendatahub.io/recommended-accelerators": (
                f'["{supported_accelerator_type.lower()}"]'
            ),
            "opendatahub.io/template-display-name": (
                f"vLLM CPU ({supported_accelerator_type}) ServingRuntime"
            ),
            "openshift.io/display-name": (
                f"vLLM CPU ({supported_accelerator_type}) ServingRuntime for KServe"
            ),
        },
        spec_annotations={
            "prometheus.io/path": "/metrics",
            "prometheus.io/port": "8080",
        },
        containers=[
            {
                "name": "kserve-container",
                "image": vllm_runtime_image,
                "command": [
                    "python",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                ],
                "args": [
                    "--port=8080",
                    "--served-model-name={{.Name}}",
                ],
                "env": [
                    {"name": "HF_HOME", "value": "/tmp/hf_home"},
                    *IBM_POWER_Z_ENV_VARIABLES,
                ],
                "ports": [
                    {
                        "containerPort": 8080,
                        "protocol": "TCP",
                    }
                ],
            }
        ],
        multi_model=False,
        supported_model_formats=[
            {
                "autoSelect": True,
                "name": "vLLM",
            }
        ],
    )

    serving_runtime.deploy()

    LOGGER.info(
        f"ServingRuntime {serving_runtime.name} created successfully"
    )

    yield serving_runtime

    serving_runtime.clean_up()


@pytest.fixture(scope="class")
def ibm_power_z_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ibm_power_z_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    vllm_model_service_account: Any,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    """vLLM InferenceService for CPU Power or Z deployments."""

    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": ibm_power_z_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": (
            ibm_power_z_serving_runtime.instance.spec.supportedModelFormats[0].name
        ),
        "model_service_account": vllm_model_service_account.name,
        "deployment_mode": request.param.get(
            "deployment_mode",
            KServeDeploymentType.STANDARD,
        ),
        "external_route": True,
        "resources": deepcopy(
            x=IBM_POWER_Z_PREDICT_RESOURCES,
        ),
        "timeout": request.param.get(
            "timeout",
            Timeout.TIMEOUT_30MIN,
        ),
    }

    if arguments := request.param.get("runtime_argument"):
        arguments = [
            arg
            for arg in arguments
            if not arg.startswith("--quantization")
        ]

        if quantization := request.param.get("quantization"):
            validate_supported_quantization_schema(
                q_type=quantization
            )
            arguments.append(
                f"--quantization={quantization}"
            )

        isvc_kwargs["argument"] = dedupe_vllm_cli_args(
            arguments=arguments
        )

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    add_image_pull_secrets_if_configured(
        isvc_kwargs=isvc_kwargs,
        kserve_registry_pull_secret=kserve_registry_pull_secret,
    )

    with create_isvc(**isvc_kwargs) as isvc:
        import time

        try:
            LOGGER.info(
                f"Waiting for Route {isvc.name} creation"
            )

            route = Route(
                client=admin_client,
                name=isvc.name,
                namespace=model_namespace.name,
            )

            for _ in range(60):
                if route.exists:
                    break

                time.sleep(2)

                route = Route(
                    client=admin_client,
                    name=isvc.name,
                    namespace=model_namespace.name,
                )

            if route.exists:
                LOGGER.info(
                    f"Found Route {route.name}, patching timeout"
                )

                # Get the current route instance as a dict
                route_dict = route.instance.to_dict()
                
                # Update annotations
                if "metadata" not in route_dict:
                    route_dict["metadata"] = {}
                if "annotations" not in route_dict["metadata"]:
                    route_dict["metadata"]["annotations"] = {}
                
                route_dict["metadata"]["annotations"][
                    "haproxy.router.openshift.io/timeout"
                ] = "300s"

                # Patch the route using the API
                route.api.patch(
                    body=route_dict,
                    name=route.name,
                    namespace=model_namespace.name,
                )

                LOGGER.info(
                    f"Route {route.name} updated with timeout=300s"
                )

                time.sleep(5)

                verify_route = Route(
                    client=admin_client,
                    name=isvc.name,
                    namespace=model_namespace.name,
                )

                timeout_value = (
                    verify_route.instance.metadata.annotations.get(
                        "haproxy.router.openshift.io/timeout"
                    )
                )

                LOGGER.info(
                    f"Verified route timeout annotation = {timeout_value}"
                )

            else:
                LOGGER.warning(
                    f"Route {isvc.name} not found"
                )

        except Exception as e:
            LOGGER.warning(
                f"Could not update route timeout: {e}"
            )

        yield isvc


@pytest.fixture
def skip_if_not_ibm_power_z_raw_deployment(
    ibm_power_z_inference_service: InferenceService,
) -> None:
    skip_if_not_deployment_mode(
        isvc=ibm_power_z_inference_service,
        deployment_types=KServeDeploymentType.RAW_DEPLOYMENT_MODES,
    )

