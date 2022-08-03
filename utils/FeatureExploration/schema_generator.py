from tfx import v1 as tfx
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.orchestration.metadata import Metadata
from tfx.types import standard_component_specs
import shutil
import os

class SchemaGenerator:
    """Class to deal with generation and running of schema pipeline"""

    def __init__(self,
                pipeline_name:str,
                pipeline_root:str,
                data_root:str,
                metadata_path:str):
        self.pipeline_name = pipeline_name
        self.pipeline_root = pipeline_root
        self.data_root     = data_root
        self.metadata_path = metadata_path

    def create_schema_pipeline(self):
        """Create a pipeline for your model"""
        # ingest the data to the pipeline
        example_gen = tfx.components.CsvExampleGen(input_base=self.data_root)

        # compute statistics from the output of example gen
        statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

        # Create Schema from the statistics
        schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'], \
                                            infer_feature_shape=True)
        
        components = [
                    example_gen,
                    statistics_gen,
                    schema_gen
        ]

        return tfx.dsl.Pipeline(
            pipeline_name=self.pipeline_name,
            pipeline_root=self.pipeline_root,
            metadata_connection_config=tfx.orchestration.metadata
                                        .sqlite_metadata_connection_config(self.metadata_path),
            components=components
        )

    def run(self):
        """ Run the Pipeline """
        tfx.orchestration.LocalDagRunner().run(self.create_schema_pipeline())

    def get_latest_artifacts(self,metadata, pipeline_name, component_id):
        """ Gets the latest artifact """
        context = metadata.store.get_context_by_type_and_name(
        'node', f'{pipeline_name}.{component_id}')
        executions = metadata.store.get_executions_by_context(context.id)
        latest_execution = max(executions,
                            key=lambda e:e.last_update_time_since_epoch)
        return execution_lib.get_artifacts_dict(metadata, latest_execution.id,
                                            [metadata_store_pb2.Event.OUTPUT])
        
    def copy_to_directory(self,path_to_directory):
        """ Copy schema.pbtxt to the given directory """
        metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(self.metadata_path)
        with Metadata(metadata_connection_config) as metadata_handler:
            schema_gen_output = self.get_latest_artifacts(metadata_handler,self.pipeline_name, 'SchemaGen')
            schema_artifacts = schema_gen_output[standard_component_specs.SCHEMA_KEY]

        SCHEMA_FILENAME = 'schema.pbtxt'
        SCHEMA_GENERATED_PATH = os.path.join(schema_artifacts[0].uri, SCHEMA_FILENAME)
        shutil.copy(SCHEMA_GENERATED_PATH,path_to_directory)
