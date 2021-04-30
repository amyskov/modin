# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from modin.data_management.utils import get_default_chunksize
from io import BytesIO
import pandas
from pandas.io.common import infer_compression
import warnings

from modin.engines.base.io import FileDispatcher
from modin.backends.pandas.parsers import find_common_type_cat


class PyarrowCSVParser:
    @staticmethod
    def parse(fname, **kwargs):
        import pyarrow as pa
        import pyarrow.csv as csv

        warnings.filterwarnings("ignore")
        num_splits = kwargs.pop("num_splits", None)
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)

        delimiter = kwargs.get("delimiter", ",")
        names = kwargs.get("names", None)
        if start is not None and end is not None:
            # pop "compression" from kwargs because bio is uncompressed
            bio = FileDispatcher.file_open(
                fname, "rb", kwargs.pop("compression", "infer")
            )
            if kwargs.get("encoding", None) is not None:
                header = b"" + bio.readline()
            else:
                header = b""
            bio.seek(start)
            to_read = header + bio.read(end - start)
            bio.close()

            table = csv.read_csv(
                BytesIO(to_read),
                parse_options=csv.ParseOptions(delimiter=delimiter),
                read_options=csv.ReadOptions(column_names=names),
            )
            chunksize = get_default_chunksize(table.num_columns, num_splits)
            chunks = [
                pa.Table.from_arrays(
                    table.columns[chunksize * i : chunksize * (i + 1)],
                    names=names[chunksize * i : chunksize * (i + 1)],
                )
                for i in range(num_splits)
            ]
            return chunks + [
                table.num_rows,
                pandas.Series(
                    [t.to_pandas_dtype() for t in table.schema.types],
                    index=table.schema.names,
                ),
            ]

    infer_compression = infer_compression

    @classmethod
    def get_dtypes(cls, dtypes_ids):
        """
        Get common for all partitions dtype for each of the columns.

        Parameters
        ----------
        dtypes_ids : list
            Array with references to the partitions dtypes objects.

        Returns
        -------
        pandas.Series
            pandas.Series where index is columns names and values are
            columns dtypes.
        """
        # import pdb; pdb.set_trace()
        return (
            pandas.concat(cls.materialize(dtypes_ids), axis=1)
            .apply(lambda row: find_common_type_cat(row.values), axis=1)
            .squeeze(axis=0)
        )
