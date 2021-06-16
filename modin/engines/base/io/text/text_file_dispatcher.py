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

"""
Module houses `TextFileDispatcher` class.

`TextFileDispatcher` contains utils for text formats files, inherits util functions for
files from `FileDispatcher` class and can be used as base class for dipatchers of SQL queries.
"""

from modin.engines.base.io.file_dispatcher import FileDispatcher
from modin.data_management.utils import compute_chunksize
import numpy as np
import warnings
import io
import os
from typing import Union, Sequence, Optional, Tuple, Callable
import pandas
from pandas.core.dtypes.common import is_list_like

from modin.config import NPartitions

ColumnNamesTypes = Tuple[Union[pandas.Index, pandas.MultiIndex, pandas.Int64Index]]


class TextFileDispatcher(FileDispatcher):
    """
    Class handles utils for reading text formats files.

    Inherits some util functions for processing files from `FileDispatcher` class.
    """

    @classmethod
    def get_path_or_buffer(cls, filepath_or_buffer):
        """
        Extract path from `filepath_or_buffer`.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_csv` function.

        Returns
        -------
        str or path object
            verified `filepath_or_buffer` parameter.

        Notes
        -----
        Given a buffer, try and extract the filepath from it so that we can
        use it without having to fall back to pandas and share file objects between
        workers. Given a filepath, return it immediately.
        """
        if isinstance(filepath_or_buffer, (io.BufferedReader, io.TextIOWrapper)):
            buffer_filepath = filepath_or_buffer.name
            if cls.file_exists(buffer_filepath):
                warnings.warn(
                    "For performance reasons, the filepath will be "
                    "used in place of the file handle passed in "
                    "to load the data"
                )
                return cls.get_path(buffer_filepath)
        return filepath_or_buffer

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        """
        Build array with partitions of `cls.frame_partition_cls` class.

        Parameters
        ----------
        partition_ids : list
                Array with references to the partitions data.
        row_lengths : list
                Partitions rows lengths.
        column_widths : list
                Number of columns in each partition.

        Returns
        -------
        np.ndarray
            array with shape equals to the shape of `partition_ids` and
            filed with partitions objects.
        """
        return np.array(
            [
                [
                    cls.frame_partition_cls(
                        partition_ids[i][j],
                        length=row_lengths[i],
                        width=column_widths[j],
                    )
                    for j in range(len(partition_ids[i]))
                ]
                for i in range(len(partition_ids))
            ]
        )

    @classmethod
    def pathlib_or_pypath(cls, filepath_or_buffer):
        """
        Check if `filepath_or_buffer` is instance of `py.path.local` or `pathlib.Path`.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_csv` function.

        Returns
        -------
        bool
            Whether or not `filepath_or_buffer` is instance of `py.path.local`
            or `pathlib.Path`.
        """
        try:
            import py

            if isinstance(filepath_or_buffer, py.path.local):
                return True
        except ImportError:  # pragma: no cover
            pass
        try:
            import pathlib

            if isinstance(filepath_or_buffer, pathlib.Path):
                return True
        except ImportError:  # pragma: no cover
            pass
        return False

    @classmethod
    def offset(
        cls,
        f,
        offset_size: int,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
    ):
        """
        Move the file offset at the specified amount of bytes.

        Parameters
        ----------
        f : file-like object
            File handle that should be used for offset movement.
        offset_size : int
            Number of bytes to read and ignore.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.

        Returns
        -------
        bool
            If file pointer reached the end of the file, but did not find
            closing quote returns `False`. `True` in any other case.
        """
        if is_quoting:
            chunk = f.read(offset_size)
            outside_quotes = not chunk.count(quotechar) % 2
        else:
            f.seek(offset_size, os.SEEK_CUR)
            outside_quotes = True

        # after we read `offset_size` bytes, we most likely break the line but
        # the modin implementation doesn't work correctly in the case, so we must
        # make sure that the line is read completely to the lineterminator,
        # which is what the `_read_rows` does
        outside_quotes, _, _ = cls._read_rows(
            f,
            nrows=1,
            quotechar=quotechar,
            is_quoting=is_quoting,
            outside_quotes=outside_quotes,
        )

        return outside_quotes

    @classmethod
    def partitioned_file(
        cls,
        f,
        num_partitions: int = None,
        nrows: int = None,
        skiprows: Union[Sequence[int], Callable, int, None] = None,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
        header_size: int = 0,
        pre_reading: int = 0,
        encoding: str = None,
    ):
        """
        Compute chunk sizes in bytes for every partition.

        Parameters
        ----------
        f : file-like object
            File handle of file to be partitioned.
        num_partitions : int, optional
            For what number of partitions split a file.
            If not specified grabs the value from `modin.config.NPartitions.get()`.
        nrows : int, optional
            Number of rows of file to read.
        skiprows : array, callable or int, optional
            Specifies rows to skip.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.
        header_size : int, default: 0
            Number of rows, that occupied by header.
        pre_reading : int, default: 0
            Number of rows between header and skipped rows, that should be read.
        encoding : str, optional
            `encoding` parameter of read_* function.

        Returns
        -------
        list
            List with next elements:
                int : partition start read byte
                int : partition end read byte
                int, array-like or callable : skiprows object aligned (adopted)
                    to the exact partition
        """
        read_rows_counter = 0
        considered_rows_counter = 0
        outside_quotes = True
        partition_skiprows = 1 if encoding is not None else None
        should_handle_skiprows = skiprows is not None and not isinstance(skiprows, int)
        if num_partitions is None:
            num_partitions = NPartitions.get() - 1 if pre_reading else NPartitions.get()
        file_size = cls.file_size(f)

        if nrows:
            partition_size = max(1, num_partitions, nrows // num_partitions)
        else:
            partition_size = max(1, num_partitions, file_size // num_partitions)

        rows_skipper = cls.rows_skipper_builder(f, quotechar, is_quoting=is_quoting)
        result = []

        read_rows_counter += rows_skipper(
            header_size, skiprows=(skiprows if should_handle_skiprows else None)
        )

        if pre_reading:
            considered_rows_counter += pre_reading
            pre_reading_start = f.tell()
            outside_quotes, read_rows, _ = cls._read_rows(
                f,
                nrows=pre_reading,
                quotechar=quotechar,
                is_quoting=is_quoting,
                outside_quotes=outside_quotes,
            )
            read_rows_counter += read_rows

            result.append((pre_reading_start, f.tell(), partition_skiprows))

            # add outside_quotes
            if is_quoting and not outside_quotes:
                warnings.warn("File has mismatched quotes")

        if skiprows is not None and isinstance(skiprows, int):
            read_rows_counter += rows_skipper(skiprows)
            skiprows = None

        start = f.tell()

        if not should_handle_skiprows and nrows is None:
            while f.tell() < file_size:
                outside_quotes = cls.offset(
                    f,
                    offset_size=partition_size,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                )
                result.append((start, f.tell(), partition_skiprows))
                start = f.tell()

                # add outside_quotes
                if is_quoting and not outside_quotes:
                    warnings.warn("File has mismatched quotes")
        else:
            read = f.tell() if nrows is None else considered_rows_counter
            read_limit = file_size if nrows is None else nrows
            while f.tell() < file_size and read < read_limit:
                if should_handle_skiprows:
                    skiprows_deploy_arg = cls.handle_skiprows(
                        skiprows=skiprows,
                        # if encoding is not None, parser will read additional line
                        # with header
                        start_row=read_rows_counter - 1
                        if encoding is not None
                        else read_rows_counter,
                        extra_skiprows=list(range(partition_skiprows))
                        if partition_skiprows
                        else None,
                    )
                    skiprows_read_rows_arg = cls.handle_skiprows(
                        skiprows=skiprows,
                        start_row=read_rows_counter,
                    )
                else:
                    rows_considered = read_rows_counter
                    skiprows_deploy_arg = (
                        partition_skiprows + skiprows
                        if skiprows is not None
                        else partition_skiprows
                    )
                    skiprows_read_rows_arg = (
                        skiprows if skiprows is not None else partition_skiprows
                    )

                if (read + partition_size) > read_limit:
                    partition_size = read_limit - read

                outside_quotes, read_rows, rows_considered = cls._read_rows(
                    f,
                    nrows=partition_size if nrows else None,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                    outside_quotes=outside_quotes,
                    max_bytes=partition_size if nrows is None else None,
                    skiprows=skiprows_read_rows_arg,
                )
                if rows_considered != 0:
                    result.append((start, f.tell(), skiprows_deploy_arg))
                start = f.tell()
                read_rows_counter += read_rows
                read = f.tell() if nrows is None else read + rows_considered

                # add outside_quotes
                if is_quoting and not outside_quotes:
                    warnings.warn("File has mismatched quotes")

        return result

    @classmethod
    def _read_rows(
        cls,
        f,
        nrows: int,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
        outside_quotes: bool = True,
        max_bytes: int = None,
        skiprows: Union[Sequence[int], Callable, None] = None,
    ):
        """
        Move the file offset at the specified amount of rows.

        Parameters
        ----------
        f : file-like object
            File handle that should be used for offset movement.
        nrows : int
            Number of rows to read.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.
        outside_quotes : bool, default: True
            Whether the file pointer is within quotes or not at the time this function is called.
        max_bytes : int, optional
            The maximum number of bytes, that can be read during function call.
        skiprows : array or callable, optional
            Specifies rows to skip.

        Returns
        -------
        bool
            If file pointer reached the end of the file, but did not find
            closing quote returns `False`. `True` in any other case.
        int
            Number of rows that was read (including skipped).
        int
            Number of rows that was "considered" (read rows excluding skipped).
        """
        if nrows is None and max_bytes is None:
            max_bytes = float("inf")

        if nrows is not None and nrows <= 0:
            return True, 0, 0

        # we need this condition to avoid unnecessary checks in `stop_condition`
        # which executes in a huge for loop
        if nrows is not None and max_bytes is None:
            stop_condition = lambda rows_read: rows_read >= nrows  # noqa (E731)
        elif nrows is not None and max_bytes is not None:
            stop_condition = (
                lambda rows_read: f.tell() >= max_bytes or rows_read >= nrows
            )  # noqa (E731)
        else:
            stop_condition = lambda rows_read: f.tell() >= max_bytes  # noqa (E731)

        if max_bytes is not None:
            max_bytes = max_bytes + f.tell()

        rows_considered = 0
        rows_read = 0

        should_handle_skiprows = skiprows is not None and not isinstance(skiprows, int)

        if should_handle_skiprows:
            skiprows_handler = cls.skiprows_handler_builder(skiprows)

        for line in f:
            if is_quoting and line.count(quotechar) % 2:
                outside_quotes = not outside_quotes
            if outside_quotes:
                if should_handle_skiprows:
                    rows_considered += next(skiprows_handler)
                else:
                    rows_considered += 1
                rows_read += 1
                if stop_condition(rows_read=rows_considered):
                    break
        # case when EOF
        if not outside_quotes:
            rows_read += 1

        return outside_quotes, rows_read, rows_considered

    @classmethod
    def handle_skiprows(
        cls,
        skiprows: Union[Sequence[int], Callable, None],
        start_row: int,
        extra_skiprows: Union[Sequence, int, None] = None,
    ) -> Union[int, Sequence, Callable]:
        """
        Handle skiprows parameter according to the reference start_row.

        Parameters
        ----------
        skiprows : array or callable
            Skiprows object that should be aligned to the start_row.
        start_row : int
            Skiprows alignement reference.
        extra_skiprows : array or int, optional
            Additional rows numbers that should be skipped (indexed
            from start_row).

        Returns
        -------
        new_skiprows : list-like, int or callable
            Skiprows object, aligned to the start_row.
        """
        if extra_skiprows is None:
            extra_skiprows = []
        elif not isinstance(extra_skiprows, (list, tuple)):
            extra_skiprows = [extra_skiprows]

        def skiprows_wrapper(n):
            return n in extra_skiprows or skiprows(n + start_row)

        if callable(skiprows):
            new_skiprows = skiprows_wrapper
        elif is_list_like(skiprows):
            start = np.searchsorted(skiprows, start_row)
            new_skiprows = np.concatenate(
                [extra_skiprows, skiprows[start:] - start_row]
            )
            if len(extra_skiprows) > 0:
                new_skiprows = np.sort(new_skiprows)
        else:
            new_skiprows = skiprows

        return new_skiprows

    @classmethod
    def skiprows_handler_builder(cls, skiprows):
        """
        Build `skiprows` parameter handler.

        Build function that will iterate over lines numbers and define whatever
        iterated line number should be skipped or not in accordance to `skiprows` type.

        Parameters
        ----------
        skiprows : array or callable
            `skiprows` parameter of read_* function.

        Returns
        -------
        obj
            Object which defines whatever next row should be skipped or not.
        """
        if callable(skiprows):

            def stepper():
                row_number = 0
                while True:
                    yield not skiprows(row_number)
                    row_number += 1

        elif is_list_like(skiprows):
            # if skiprows is an array, elements of skiprows
            # will compared with increased on each step row_number.
            # If match of row_number and skiprows element is occured,
            # 0 is yielded and next element of skiprows will be compared
            # further (skiprows should be sorted).
            def stepper():
                row_number = 0
                index_to_compare = 0
                while index_to_compare < len(skiprows):
                    if skiprows[index_to_compare] == row_number:
                        index_to_compare += 1
                        yield 0
                    else:
                        yield 1
                    row_number += 1
                while True:
                    yield 1

        else:

            def stepper():
                while True:
                    yield 1

        return stepper()

    # _read helper functions
    @classmethod
    def rows_skipper_builder(cls, f, quotechar, is_quoting, skiprows=None):
        """
        Build object for skipping passed number of lines.

        Parameters
        ----------
        f : file-like object
            File handle that should be used for offset movement.
        quotechar : bytes
            Indicate quote in a file.
        is_quoting : bool
            Whether or not to consider quotes.
        skiprows : array or callable, optional
            Specifies rows to skip.

        Returns
        -------
        object
            skipper object.
        """
        _skiprows = skiprows

        def skipper(n, skiprows=_skiprows):
            return cls._read_rows(
                f,
                quotechar=quotechar,
                is_quoting=is_quoting,
                nrows=n,
                skiprows=skiprows,
            )[1]

        return skipper

    @classmethod
    def _define_header_size(
        cls,
        header: Union[int, Sequence[int], str, None] = "infer",
        names: Optional[Sequence] = None,
    ) -> int:
        """
        Define the number of rows that are used by header.

        Parameters
        ----------
        header : int, list of int or str, default: "infer"
            Original `header` parameter of `read_csv` function.
        names :  array-like, optional
            Original names parameter of `read_csv` function.

        Returns
        -------
        header_size : int
            The number of rows that are used by header.
        """
        header_size = 0
        if header == "infer" and names is None:
            header_size += 1
        elif isinstance(header, int):
            header_size += header + 1
        elif hasattr(header, "__iter__") and not isinstance(header, str):
            header_size += max(header) + 1

        return header_size

    @classmethod
    def _define_metadata(
        cls,
        df: pandas.DataFrame,
        num_splits: int,
        column_names: ColumnNamesTypes,
    ) -> Tuple[list, int]:
        """
        Define partitioning metadata.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to split.
        num_splits : int
            The maximum number of splits to separate the DataFrame into.
        column_names : ColumnNamesTypes
            Column names of df.

        Returns
        -------
        column_widths : list
            Column width to use during new frame creation (number of
            columns for each partition).
        num_splits : int
            Updated `num_splits` parameter.
        """
        column_chunksize = compute_chunksize(df, num_splits, axis=1)
        if column_chunksize > len(column_names):
            column_widths = [len(column_names)]
            # This prevents us from unnecessarily serializing a bunch of empty
            # objects.
            num_splits = 1
        else:
            # split columns into chunks with maximal size column_chunksize, for example
            # if num_splits == 4, len(column_names) == 80 and column_chunksize == 32,
            # column_widths will be [32, 32, 16, 0]
            column_widths = []
            for i in range(num_splits):
                if len(column_names) > (column_chunksize * i):
                    if len(column_names) > (column_chunksize * (i + 1)):
                        column_widths.append(column_chunksize)
                    else:
                        column_widths.append(len(column_names) - (column_chunksize * i))
                else:
                    column_widths.append(0)

        return column_widths, num_splits

    @classmethod
    def _launch_tasks(cls, splits: list, **partition_kwargs) -> Tuple[list, list, list]:
        """
        Launch tasks to read partitions.

        Parameters
        ----------
        splits : list
            List of tuples with partitions data, which defines
            parser task (start/end read bytes and etc.).
        **partition_kwargs : dict
            `kwargs` that should be passed to the parser function.

        Returns
        -------
        partition_ids : list
            array with references to the partitions data.
        index_ids : list
            array with references to the partitions index objects.
        dtypes_ids : list
            array with references to the partitions dtypes objects.
        """
        partition_ids = []
        index_ids = []
        dtypes_ids = []
        for split_data in splits:
            partition_kwargs.update(
                {
                    "start": split_data[0],
                    "end": split_data[1],
                    "skiprows": split_data[2],
                }
            )
            partition_id = cls.deploy(
                cls.parse, partition_kwargs.get("num_splits") + 2, partition_kwargs
            )
            partition_ids.append(partition_id[:-2])
            index_ids.append(partition_id[-2])
            dtypes_ids.append(partition_id[-1])

        return partition_ids, index_ids, dtypes_ids
