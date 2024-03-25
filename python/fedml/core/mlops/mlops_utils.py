import os
from os.path import expanduser
import time
import json
from typing import Dict, Optional, Any

import yaml
from dataclasses import dataclass, asdict


class MLOpsUtils:
    _ntp_offset = None
    BYTES_TO_GB = 1 / (1024 * 1024 * 1024)

    @staticmethod
    def calc_ntp_from_config(mlops_config):
        if mlops_config is None:
            return

        ntp_response = mlops_config.get("NTP_RESPONSE", None)
        if ntp_response is None:
            return

        # setup ntp time from the configs
        device_recv_time = int(time.time() * 1000)
        device_send_time = ntp_response.get("deviceSendTime", None)
        server_recv_time = ntp_response.get("serverRecvTime", None)
        server_send_time = ntp_response.get("serverSendTime", None)
        if device_send_time is None or server_recv_time is None or server_send_time is None:
            return

        # calculate the time offset(int)
        ntp_time = (server_recv_time + server_send_time + device_recv_time - device_send_time) // 2
        ntp_offset = ntp_time - device_recv_time

        # set the time offset
        MLOpsUtils.set_ntp_offset(ntp_offset)

    @staticmethod
    def set_ntp_offset(ntp_offset):
        MLOpsUtils._ntp_offset = ntp_offset

    @staticmethod
    def get_ntp_time():
        if MLOpsUtils._ntp_offset is not None:
            return int(time.time() * 1000) + MLOpsUtils._ntp_offset
        return int(time.time() * 1000)

    @staticmethod
    def get_ntp_offset():
        return MLOpsUtils._ntp_offset

    @staticmethod
    def write_log_trace(log_trace):
        log_trace_dir = os.path.join(expanduser("~"), "fedml_log")
        if not os.path.exists(log_trace_dir):
            os.makedirs(log_trace_dir, exist_ok=True)

        log_file_obj = open(os.path.join(log_trace_dir, "logs.txt"), "a")
        log_file_obj.write("{}\n".format(log_trace))
        log_file_obj.close()


@dataclass
class LogFile:
    file_name: str
    uploaded_file_index: int = 0
    upload_complete: bool = False


class MLOpsLoggingUtils:
    LOG_CONFIG_FILE = "log_config.yaml"

    @staticmethod
    def build_log_file_path_with_run_params(
            run_id, edge_id, log_file_dir, is_server=False, log_file_prefix=None
    ):
        program_prefix = "FedML-{} @device-id-{}".format(
            "Server" if is_server else "Client", edge_id)
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(
            log_file_dir, "fedml-run{}-{}-edge-{}.log".format(
                "" if log_file_prefix is None else f"-{log_file_prefix}", run_id, edge_id
            ))

        return log_file_path, program_prefix

    @staticmethod
    def build_log_file_path(in_args):
        if in_args.role == "server":
            if hasattr(in_args, "server_id"):
                edge_id = in_args.server_id
            else:
                if hasattr(in_args, "edge_id"):
                    edge_id = in_args.edge_id
                else:
                    edge_id = 0
            program_prefix = "FedML-Server @device-id-{}".format(edge_id)
        else:
            if hasattr(in_args, "client_id"):
                edge_id = in_args.client_id
            elif hasattr(in_args, "client_id_list"):
                if in_args.client_id_list is None:
                    edge_id = 0
                else:
                    edge_ids = json.loads(in_args.client_id_list)
                    if len(edge_ids) > 0:
                        edge_id = edge_ids[0]
                    else:
                        edge_id = 0
            else:
                if hasattr(in_args, "edge_id"):
                    edge_id = in_args.edge_id
                else:
                    edge_id = 0
            program_prefix = "FedML-Client @device-id-{edge}".format(edge=edge_id)

        if not os.path.exists(in_args.log_file_dir):
            os.makedirs(in_args.log_file_dir, exist_ok=True)
        log_file_path = os.path.join(in_args.log_file_dir, "fedml-run-"
                                     + str(in_args.run_id)
                                     + "-edge-"
                                     + str(edge_id)
                                     + ".log")

        return log_file_path, program_prefix

    @staticmethod
    def load_log_config(run_id, device_id, log_config_file) -> Dict[str, LogFile]:
        try:
            log_config_key = "log_config_{}_{}".format(run_id, device_id)
            log_config = MLOpsLoggingUtils.load_yaml_config(log_config_file)
            run_log_config = log_config.get(log_config_key, {})
            config_data = {}
            for index, data in run_log_config.items():
                config_data[index] = LogFile(**data)
            return config_data
            #
            #
            # return log_config[log_config_key]
            # # current_rotate_count = log_config[log_config_key].get("file_rotate_count", 0)
            # # self.file_rotate_count = max(self.file_rotate_count, current_rotate_count)
            # # if self.file_rotate_count > current_rotate_count:
            # #     self.log_line_index = 0
            # # else:
            # #     self.log_line_index = self.log_config[log_config_key]["log_line_index"]
        except Exception as e:
            raise ValueError("Error loading log config: {}".format(e))

    @staticmethod
    def save_log_config(run_id, device_id, log_config_file, config_data):
        try:
            log_config_key = "log_config_{}_{}".format(run_id, device_id)
            log_config = MLOpsLoggingUtils.load_yaml_config(log_config_file)
            log_config[log_config_key] = MLOpsLoggingUtils.__convert_to_dict(config_data)
            with open(log_config_file, "w") as stream:
                yaml.dump(log_config, stream)
        except Exception as e:
            raise ValueError("Error saving log config: {}".format(e))

    @staticmethod
    def load_yaml_config(log_config_file):
        """Helper function to load a yaml config file"""
        if not os.path.exists(log_config_file):
            MLOpsLoggingUtils.generate_yaml_doc({}, log_config_file)
        with open(log_config_file, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Yaml error - check yaml file")

    # @staticmethod
    # def get_id_from_filename(run_id, device_id, filename, log_config_file) -> Optional[str]:
    #     config_data = MLOpsLoggingUtils.load_log_config(run_id, device_id, log_config_file)
    #     for id, data in config_data.items():
    #         if data.file_name == filename:
    #             return id
    #     return None

    @staticmethod
    def generate_yaml_doc(log_config_object, yaml_file):
        try:
            file = open(yaml_file, "w", encoding="utf-8")
            yaml.dump(log_config_object, file)
            file.close()
        except Exception as e:
            pass

    @staticmethod
    def __convert_to_dict(obj: Any) -> Any:
        if isinstance(obj, LogFile):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: MLOpsLoggingUtils.__convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MLOpsLoggingUtils.__convert_to_dict(item) for item in obj]
        else:
            return obj
