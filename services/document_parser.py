"""
Responsible for converting uploaded files into LangChain Documents.

Processing strategy:

Text files
    PDF / DOCX
        -> Extract text
        -> Split into chunks
        -> LangChain Documents

Tabular files
    CSV / XLS / XLSX
        -> One Document per row
        -> Row values stored as metadata for filtering
"""

import os

import fitz
import pandas as pd
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


class DocumentProcessor:
    """
    Convert uploaded files into LangChain Documents.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".csv",
        ".xls",
        ".xlsx",
    }

    def process(self, file, metadata=None):
        """
        Convert an uploaded file into LangChain Documents.
        """

        metadata = metadata or {}

        extension = os.path.splitext(file.filename)[1].lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        if extension == ".pdf":
            return self._process_pdf(file.file, metadata)

        if extension == ".docx":
            return self._process_docx(file.file, metadata)

        if extension == ".csv":
            return self._process_csv(file.file, metadata)

        return self._process_excel(file.file, metadata)

    # =====================================================
    # PDF
    # =====================================================

    def _process_pdf(self, file, metadata):
        file.seek(0)

        pdf = fitz.open(
            stream=file.read(),
            filetype="pdf",
        )

        text = "\n".join(
            page.get_text()
            for page in pdf
        )

        if not text.strip():
            raise ValueError(
                "Unable to extract text from PDF."
            )

        return self._create_chunked_documents(
            text=text,
            metadata=metadata,
        )

    # =====================================================
    # DOCX
    # =====================================================

    def _process_docx(self, file, metadata):
        file.seek(0)

        document = DocxDocument(file)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        if not text.strip():
            raise ValueError(
                "Unable to extract text from DOCX."
            )

        return self._create_chunked_documents(
            text=text,
            metadata=metadata,
        )

    # =====================================================
    # CSV
    # =====================================================

    def _process_csv(self, file, metadata):
        file.seek(0)

        try:
            dataframe = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            dataframe = pd.read_csv(
                file,
                encoding="latin-1",
            )

        return self._create_row_documents(
            dataframe=dataframe,
            metadata=metadata,
        )

    # =====================================================
    # Excel
    # =====================================================

    def _process_excel(self, file, metadata):
        file.seek(0)

        sheets = pd.read_excel(
            file,
            sheet_name=None,
        )

        documents = []

        for sheet_name, dataframe in sheets.items():

            documents.extend(
                self._create_row_documents(
                    dataframe=dataframe,
                    metadata={
                        **metadata,
                        "sheet_name": sheet_name,
                    },
                )
            )

        return documents

    # =====================================================
    # Helpers
    # =====================================================

    def _create_chunked_documents(
        self,
        text,
        metadata,
    ):
        """
        Split extracted text into chunked LangChain Documents.
        """

        return TEXT_SPLITTER.split_documents(
            [
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            ]
        )

    def _create_row_documents(
        self,
        dataframe,
        metadata,
    ):
        """
        Create one LangChain Document per DataFrame row.

        - page_content is used for semantic search.
        - metadata stores structured values for filtering.
        """

        documents = []

        for _, row in dataframe.iterrows():

            # Ignore completely empty rows.
            if row.isna().all():
                continue

            row_metadata = {
                column: self._normalize_metadata_value(value)
                for column, value in row.items()
            }

            row_text = "\n".join(
                f"{column}: {value}"
                for column, value in row_metadata.items()
            )

            documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        **metadata,
                        **row_metadata,
                    },
                )
            )

        return documents

    @staticmethod
    def _normalize_metadata_value(value):
        """
        Convert Pandas/NumPy values into native Python types.
        """

        if pd.isna(value):
            return None

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, float) and value.is_integer():
            return int(value)

        return value
