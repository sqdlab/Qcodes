import pytest
import numpy as np

from qcodes.dataset.sqlite_base import connect
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.database import path_of_connection
from qcodes.dataset.database_copy_paste import copy_runs_into_db
from qcodes.tests.dataset.temporary_databases import two_empty_temp_db_connections
from qcodes.tests.dataset.test_descriptions import some_paramspecs
from qcodes.tests.dataset.test_database_creation_and_upgrading import error_caused_by


def test_basic_copy_paste(two_empty_temp_db_connections, some_paramspecs):
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_of_connection(source_conn)
    target_path = path_of_connection(target_conn)

    type_casters = {'numeric': float,
                    'array': (lambda x: np.array(x) if hasattr(x, '__iter__')
                              else np.array([x])),
                    'text': str}

    source_exp = Experiment(conn=source_conn)
    source_dataset = DataSet(conn=source_conn)

    with pytest.raises(RuntimeError) as excinfo:
        copy_runs_into_db(source_path, target_path, source_dataset.run_id)

    assert error_caused_by(excinfo, 'Dataset not completed')

    for ps in some_paramspecs[1].values():
        source_dataset.add_parameter(ps)

    for value in range(10):
        result = {ps.name: type_casters[ps.type](value)
                  for ps in some_paramspecs[1].values()}
        source_dataset.add_result(result)

    source_dataset.mark_complete()

    copy_runs_into_db(source_path, target_path, source_dataset.run_id)

    target_exp = Experiment(conn=target_conn, exp_id=1)

    length1 = len(target_exp)

    # trying to insert the same run again should be a NOOP
    copy_runs_into_db(source_path, target_path, source_dataset.run_id)

    assert len(target_exp) == length1

    target_dataset = DataSet(conn=source_conn, run_id=1)

    # Now make the interesting comparisons: are the target objects the same as
    # the source objects?

    exp_attrs = ['name', 'sample_name', 'format_string', 'started_at',
                 'finished_at']

    ds_attrs = ['name', 'table_name', 'guid', 'number_of_results',
                'counter', 'parameters', 'paramspecs', 'exp_name',
                'sample_name', 'completed', 'snapshot', 'run_timestamp_raw']

    for ds_attr in ds_attrs:
        assert getattr(source_dataset, ds_attr) == getattr(target_dataset, ds_attr)

    for exp_attr in exp_attrs:
        assert getattr(source_exp, exp_attr) == getattr(target_exp, exp_attr)

    source_data = source_dataset.get_data(*source_dataset.parameters.split(','))
    target_data = target_dataset.get_data(*target_dataset.parameters.split(','))

    assert source_data == target_data


