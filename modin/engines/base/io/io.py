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
Module houses `BaseIO` class.

`BaseIO` is base class for IO classes, that stores IO functions.
"""

import pandas
from pandas.util._decorators import doc
from collections import OrderedDict
from modin.error_message import ErrorMessage
from modin.backends.base.query_compiler import BaseQueryCompiler
from typing import Optional

# TODO (amyskov): replace `For parameters description please refer to pandas API.` statement with
# @_inherit_docstrings decorator when #2969 will be merged.
_doc_default_io_method = """
{summary} using pandas.

For parameters description please refer to pandas API.

Returns
-------
{returns}
"""

_doc_returns_qc = """BaseQueryCompiler
    QueryCompiler with read data."""

_doc_returns_qc_or_parser = """BaseQueryCompiler or TextParser
    QueryCompiler or TextParser with read data."""


class BaseIO(object):
    """Class for basic utils and default implementation of IO functions."""

    query_compiler_cls: BaseQueryCompiler = None
    frame_cls = None

    @classmethod
    def from_non_pandas(cls, *args, **kwargs):
        """
        Improve non-pandas object to an advanced Modin query compiler.

        Parameters
        ----------
        *args : iterable
            Positional arguments to be passed into `func`.
        **kwargs : dict
            Keyword arguments to be passed into `func`.
        """
        return None

    @classmethod
    def from_pandas(cls, df):
        """
        Improve simple pandas DataFrame to an advanced Modin query compiler.

        Parameters
        ----------
        df : pandas.DataFrame
            The pandas DataFrame to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the pandas DataFrame.
        """
        return cls.query_compiler_cls.from_pandas(df, cls.frame_cls)

    @classmethod
    def from_arrow(cls, at):
        """
        Improve simple Arrow Table to an advanced Modin query compiler.

        Parameters
        ----------
        at : Arrow Table
            The Arrow Table to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Arrow Table.
        """
        return cls.query_compiler_cls.from_arrow(at, cls.frame_cls)

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Load a parquet object from the file path, returning a query compiler",
        returns=_doc_returns_qc,
    )
    def read_parquet(cls, path, engine, columns, use_nullable_dtypes, **kwargs):
        ErrorMessage.default_to_pandas("`read_parquet`")
        return cls.from_pandas(
            pandas.read_parquet(path, engine, columns, use_nullable_dtypes, **kwargs)
        )

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read a comma-separated values (CSV) file into query compiler",
        returns=_doc_returns_qc_or_parser,
    )
    def read_csv(
        cls,
        filepath_or_buffer,
        sep=",",
        delimiter=None,
        header="infer",
        names=None,
        index_col=None,
        usecols=None,
        squeeze=False,
        prefix=None,
        mangle_dupe_cols=True,
        dtype=None,
        engine=None,
        converters=None,
        true_values=None,
        false_values=None,
        skipinitialspace=False,
        skiprows=None,
        nrows=None,
        na_values=None,
        keep_default_na=True,
        na_filter=True,
        verbose=False,
        skip_blank_lines=True,
        parse_dates=False,
        infer_datetime_format=False,
        keep_date_col=False,
        date_parser=None,
        dayfirst=False,
        cache_dates=True,
        iterator=False,
        chunksize=None,
        compression="infer",
        thousands=None,
        decimal=b".",
        lineterminator=None,
        quotechar='"',
        quoting=0,
        escapechar=None,
        comment=None,
        encoding=None,
        dialect=None,
        error_bad_lines=True,
        warn_bad_lines=True,
        skipfooter=0,
        doublequote=True,
        delim_whitespace=False,
        low_memory=True,
        memory_map=False,
        float_precision=None,
        storage_options=None,
    ):
        kwargs = {
            "filepath_or_buffer": filepath_or_buffer,
            "sep": sep,
            "delimiter": delimiter,
            "header": header,
            "names": names,
            "index_col": index_col,
            "usecols": usecols,
            "squeeze": squeeze,
            "prefix": prefix,
            "mangle_dupe_cols": mangle_dupe_cols,
            "dtype": dtype,
            "engine": engine,
            "converters": converters,
            "true_values": true_values,
            "false_values": false_values,
            "skipinitialspace": skipinitialspace,
            "skiprows": skiprows,
            "nrows": nrows,
            "na_values": na_values,
            "keep_default_na": keep_default_na,
            "na_filter": na_filter,
            "verbose": verbose,
            "skip_blank_lines": skip_blank_lines,
            "parse_dates": parse_dates,
            "infer_datetime_format": infer_datetime_format,
            "keep_date_col": keep_date_col,
            "date_parser": date_parser,
            "dayfirst": dayfirst,
            "cache_dates": cache_dates,
            "iterator": iterator,
            "chunksize": chunksize,
            "compression": compression,
            "thousands": thousands,
            "decimal": decimal,
            "lineterminator": lineterminator,
            "quotechar": quotechar,
            "quoting": quoting,
            "escapechar": escapechar,
            "comment": comment,
            "encoding": encoding,
            "dialect": dialect,
            "error_bad_lines": error_bad_lines,
            "warn_bad_lines": warn_bad_lines,
            "skipfooter": skipfooter,
            "doublequote": doublequote,
            "delim_whitespace": delim_whitespace,
            "low_memory": low_memory,
            "memory_map": memory_map,
            "float_precision": float_precision,
            "storage_options": storage_options,
        }
        ErrorMessage.default_to_pandas("`read_csv`")
        return cls._read(**kwargs)

    @classmethod
    def _read(cls, **kwargs):
        """
        Read csv file into query compiler.

        Parameters
        ----------
        **kwargs:
            `read_csv` function kwargs including `filepath_or_buffer` parameter.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with read data.
        """
        pd_obj = pandas.read_csv(**kwargs)
        if isinstance(pd_obj, pandas.DataFrame):
            return cls.from_pandas(pd_obj)
        if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
            # Overwriting the read method should return a Modin DataFrame for calls
            # to __next__ and get_chunk
            pd_read = pd_obj.read
            pd_obj.read = lambda *args, **kwargs: cls.from_pandas(
                pd_read(*args, **kwargs)
            )
        return pd_obj

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Convert a JSON string to query compiler",
        returns=_doc_returns_qc,
    )
    def read_json(
        cls,
        path_or_buf=None,
        orient=None,
        typ="frame",
        dtype=True,
        convert_axes=True,
        convert_dates=True,
        keep_default_dates=True,
        numpy=False,
        precise_float=False,
        date_unit=None,
        encoding=None,
        lines=False,
        chunksize=None,
        compression="infer",
        nrows: Optional[int] = None,
        storage_options=None,
    ):
        ErrorMessage.default_to_pandas("`read_json`")
        kwargs = {
            "path_or_buf": path_or_buf,
            "orient": orient,
            "typ": typ,
            "dtype": dtype,
            "convert_axes": convert_axes,
            "convert_dates": convert_dates,
            "keep_default_dates": keep_default_dates,
            "numpy": numpy,
            "precise_float": precise_float,
            "date_unit": date_unit,
            "encoding": encoding,
            "lines": lines,
            "chunksize": chunksize,
            "compression": compression,
            "nrows": nrows,
            "storage_options": storage_options,
        }
        return cls.from_pandas(pandas.read_json(**kwargs))

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Load data from Google BigQuery into query compiler",
        returns=_doc_returns_qc,
    )
    def read_gbq(
        cls,
        query: str,
        project_id=None,
        index_col=None,
        col_order=None,
        reauth=False,
        auth_local_webserver=False,
        dialect=None,
        location=None,
        configuration=None,
        credentials=None,
        use_bqstorage_api=None,
        private_key=None,
        verbose=None,
        progress_bar_type=None,
        max_results=None,
    ):
        ErrorMessage.default_to_pandas("`read_gbq`")
        return cls.from_pandas(
            pandas.read_gbq(
                query,
                project_id=project_id,
                index_col=index_col,
                col_order=col_order,
                reauth=reauth,
                auth_local_webserver=auth_local_webserver,
                dialect=dialect,
                location=location,
                configuration=configuration,
                credentials=credentials,
                use_bqstorage_api=use_bqstorage_api,
                private_key=private_key,
                verbose=verbose,
                progress_bar_type=progress_bar_type,
                max_results=max_results,
            )
        )

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read HTML tables into query compiler",
        returns=_doc_returns_qc,
    )
    def read_html(
        cls,
        io,
        match=".+",
        flavor=None,
        header=None,
        index_col=None,
        skiprows=None,
        attrs=None,
        parse_dates=False,
        thousands=",",
        encoding=None,
        decimal=".",
        converters=None,
        na_values=None,
        keep_default_na=True,
        displayed_only=True,
    ):
        ErrorMessage.default_to_pandas("`read_html`")
        kwargs = {
            "io": io,
            "match": match,
            "flavor": flavor,
            "header": header,
            "index_col": index_col,
            "skiprows": skiprows,
            "attrs": attrs,
            "parse_dates": parse_dates,
            "thousands": thousands,
            "encoding": encoding,
            "decimal": decimal,
            "converters": converters,
            "na_values": na_values,
            "keep_default_na": keep_default_na,
            "displayed_only": displayed_only,
        }
        return cls.from_pandas(pandas.read_html(**kwargs)[0])

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read text from clipboard into query compiler",
        returns=_doc_returns_qc,
    )
    def read_clipboard(cls, sep=r"\s+", **kwargs):  # pragma: no cover
        ErrorMessage.default_to_pandas("`read_clipboard`")
        return cls.from_pandas(pandas.read_clipboard(sep=sep, **kwargs))

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read an Excel file into query compiler",
        returns="""BaseQueryCompiler or dict/OrderedDict :
    QueryCompiler or OrderedDict/dict with read data.""",
    )
    def read_excel(
        cls,
        io,
        sheet_name=0,
        header=0,
        names=None,
        index_col=None,
        usecols=None,
        squeeze=False,
        dtype=None,
        engine=None,
        converters=None,
        true_values=None,
        false_values=None,
        skiprows=None,
        nrows=None,
        na_values=None,
        keep_default_na=True,
        verbose=False,
        parse_dates=False,
        date_parser=None,
        thousands=None,
        comment=None,
        skip_footer=0,
        skipfooter=0,
        convert_float=True,
        mangle_dupe_cols=True,
        na_filter=True,
        **kwds,
    ):
        if skip_footer != 0:
            skipfooter = skip_footer
        ErrorMessage.default_to_pandas("`read_excel`")
        intermediate = pandas.read_excel(
            io,
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            squeeze=squeeze,
            dtype=dtype,
            engine=engine,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            verbose=verbose,
            parse_dates=parse_dates,
            date_parser=date_parser,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            convert_float=convert_float,
            mangle_dupe_cols=mangle_dupe_cols,
            na_filter=na_filter,
            **kwds,
        )
        if isinstance(intermediate, (OrderedDict, dict)):
            parsed = type(intermediate)()
            for key in intermediate.keys():
                parsed[key] = cls.from_pandas(intermediate.get(key))
            return parsed
        else:
            return cls.from_pandas(intermediate)

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read data from hdf store into query compiler",
        returns=_doc_returns_qc,
    )
    def read_hdf(
        cls,
        path_or_buf,
        key=None,
        mode: str = "r",
        errors: str = "strict",
        where=None,
        start=None,
        stop=None,
        columns=None,
        iterator=False,
        chunksize=None,
        **kwargs,
    ):
        ErrorMessage.default_to_pandas("`read_hdf`")
        return cls.from_pandas(
            pandas.read_hdf(
                path_or_buf,
                key=key,
                mode=mode,
                columns=columns,
                errors=errors,
                where=where,
                start=start,
                stop=stop,
                iterator=iterator,
                chunksize=chunksize,
                **kwargs,
            )
        )

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Load a feather-format object from the file path into query compiler",
        returns=_doc_returns_qc,
    )
    def read_feather(cls, path, columns=None, use_threads=True, storage_options=None):
        ErrorMessage.default_to_pandas("`read_feather`")
        return cls.from_pandas(
            pandas.read_feather(
                path,
                columns=columns,
                use_threads=use_threads,
                storage_options=storage_options,
            )
        )

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read Stata file into query compiler",
        returns=_doc_returns_qc,
    )
    def read_stata(
        cls,
        filepath_or_buffer,
        convert_dates=True,
        convert_categoricals=True,
        index_col=None,
        convert_missing=False,
        preserve_dtypes=True,
        columns=None,
        order_categoricals=True,
        chunksize=None,
        iterator=False,
        storage_options=None,
    ):
        ErrorMessage.default_to_pandas("`read_stata`")
        kwargs = {
            "filepath_or_buffer": filepath_or_buffer,
            "convert_dates": convert_dates,
            "convert_categoricals": convert_categoricals,
            "index_col": index_col,
            "convert_missing": convert_missing,
            "preserve_dtypes": preserve_dtypes,
            "columns": columns,
            "order_categoricals": order_categoricals,
            "chunksize": chunksize,
            "iterator": iterator,
            "storage_options": storage_options,
        }
        return cls.from_pandas(pandas.read_stata(**kwargs))

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read SAS files stored as either XPORT or SAS7BDAT format files\ninto query compiler",
        returns=_doc_returns_qc,
    )
    def read_sas(
        cls,
        filepath_or_buffer,
        format=None,
        index=None,
        encoding=None,
        chunksize=None,
        iterator=False,
    ):  # pragma: no cover
        ErrorMessage.default_to_pandas("`read_sas`")
        return cls.from_pandas(
            pandas.read_sas(
                filepath_or_buffer,
                format=format,
                index=index,
                encoding=encoding,
                chunksize=chunksize,
                iterator=iterator,
            )
        )

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Load pickled pandas object (or any object) from file into query compiler",
        returns=_doc_returns_qc,
    )
    def read_pickle(cls, filepath_or_buffer, compression="infer", storage_options=None):
        ErrorMessage.default_to_pandas("`read_pickle`")
        return cls.from_pandas(
            pandas.read_pickle(
                filepath_or_buffer,
                compression=compression,
                storage_options=storage_options,
            )
        )

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read SQL query or database table into query compiler",
        returns=_doc_returns_qc,
    )
    def read_sql(
        cls,
        sql,
        con,
        index_col=None,
        coerce_float=True,
        params=None,
        parse_dates=None,
        columns=None,
        chunksize=None,
    ):
        ErrorMessage.default_to_pandas("`read_sql`")
        return cls.from_pandas(
            pandas.read_sql(
                sql,
                con,
                index_col=index_col,
                coerce_float=coerce_float,
                params=params,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
            )
        )

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read a table of fixed-width formatted lines into query compiler",
        returns=_doc_returns_qc_or_parser,
    )
    def read_fwf(
        cls, filepath_or_buffer, colspecs="infer", widths=None, infer_nrows=100, **kwds
    ):
        ErrorMessage.default_to_pandas("`read_fwf`")
        pd_obj = pandas.read_fwf(
            filepath_or_buffer,
            colspecs=colspecs,
            widths=widths,
            infer_nrows=infer_nrows,
            **kwds,
        )
        if isinstance(pd_obj, pandas.DataFrame):
            return cls.from_pandas(pd_obj)
        if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
            # Overwriting the read method should return a Modin DataFrame for calls
            # to __next__ and get_chunk
            pd_read = pd_obj.read
            pd_obj.read = lambda *args, **kwargs: cls.from_pandas(
                pd_read(*args, **kwargs)
            )
        return pd_obj

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read SQL database table into query compiler",
        returns=_doc_returns_qc,
    )
    def read_sql_table(
        cls,
        table_name,
        con,
        schema=None,
        index_col=None,
        coerce_float=True,
        parse_dates=None,
        columns=None,
        chunksize=None,
    ):
        ErrorMessage.default_to_pandas("`read_sql_table`")
        return cls.from_pandas(
            pandas.read_sql_table(
                table_name,
                con,
                schema=schema,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
            )
        )

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Read SQL query into query compiler",
        returns=_doc_returns_qc,
    )
    def read_sql_query(
        cls,
        sql,
        con,
        index_col=None,
        coerce_float=True,
        params=None,
        parse_dates=None,
        chunksize=None,
    ):
        ErrorMessage.default_to_pandas("`read_sql_query`")
        return cls.from_pandas(
            pandas.read_sql_query(
                sql,
                con,
                index_col=index_col,
                coerce_float=coerce_float,
                params=params,
                parse_dates=parse_dates,
                chunksize=chunksize,
            )
        )

    @classmethod
    @doc(
        _doc_default_io_method,
        summary="Load an SPSS file from the file path, returning a query compiler",
        returns=_doc_returns_qc,
    )
    def read_spss(cls, path, usecols, convert_categoricals):
        ErrorMessage.default_to_pandas("`read_spss`")
        return cls.from_pandas(pandas.read_spss(path, usecols, convert_categoricals))

    @classmethod
    def to_sql(
        cls,
        qc,
        name,
        con,
        schema=None,
        if_exists="fail",
        index=True,
        index_label=None,
        chunksize=None,
        dtype=None,
        method=None,
    ):
        """
        Write records stored in a DataFrame to a SQL database using pandas.

        For parameters description please refer to pandas API.
        """
        ErrorMessage.default_to_pandas("`to_sql`")
        df = qc.to_pandas()
        df.to_sql(
            name=name,
            con=con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
        )

    @classmethod
    def to_pickle(cls, obj, path, compression="infer", protocol=4):
        """
        Pickle (serialize) object to file using pandas.

        For parameters description please refer to pandas API.
        """
        if protocol == 4:
            protocol = -1
        ErrorMessage.default_to_pandas("`to_pickle`")
        if isinstance(obj, BaseQueryCompiler):
            return pandas.to_pickle(
                obj.to_pandas(), path, compression=compression, protocol=protocol
            )
        else:
            return pandas.to_pickle(
                obj, path, compression=compression, protocol=protocol
            )

    @classmethod
    def to_csv(cls, obj, **kwargs):
        """
        Write object to a comma-separated values (CSV) file using pandas.

        For parameters description please refer to pandas API.
        """
        ErrorMessage.default_to_pandas("`to_csv`")
        if isinstance(obj, BaseQueryCompiler):
            obj = obj.to_pandas()

        return obj.to_csv(**kwargs)
