"""
responsible for:
    File
     ↓
    LangChain Documents
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

    Supported file types:

    Text files:
        - PDF
        - DOCX

    Tabular files:
        - CSV
        - XLS
        - XLSX

    Processing strategy:

    PDF / DOCX
        File -> Text -> Chunked Documents

    CSV / XLS / XLSX
        File -> Row Documents

    Output:
        List[Document]
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
        Main entry point.

        Args:
            file: Uploaded file object
            metadata: dict

        Returns:
            List[Document]
        """

        metadata = metadata or {}

        filename = file.filename
        extension = os.path.splitext(filename)[1].lower()

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

        doc = DocxDocument(file)

        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
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
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(
                file,
                encoding="latin-1",
            )

        return self._create_row_documents(
            dataframe=df,
            metadata=metadata,
        )

    # =====================================================
    # EXCEL
    # =====================================================

    def _process_excel(self, file, metadata):
        file.seek(0)
        sheets = pd.read_excel(
            file,
            sheet_name=None,
        )

        documents = []

        for sheet_name, dataframe in sheets.items():
            sheet_documents = self._create_row_documents(
                dataframe=dataframe,
                metadata={
                    **metadata,
                    "sheet_name": sheet_name,
                },
            )

            documents.extend(sheet_documents)

        return documents

    # =====================================================
    # HELPERS
    # =====================================================

    def _create_chunked_documents(
            self,
            text,
            metadata,
    ):
        """
        Create chunked documents for text-based files.
        """

        document = Document(
            page_content=text,
            metadata=metadata,
        )

        return TEXT_SPLITTER.split_documents(
            [document]
        )

    def _create_row_documents(
            self,
            dataframe,
            metadata,
    ):
        """
        Create one Document per row.

        Improves retrieval quality for structured data.
        """

        documents = []

        for row_number, (_, row) in enumerate(
                dataframe.iterrows(),
                start=1,
        ):
            row_text = "\n".join(
                f"{column}: {value}"
                for column, value in row.items()
            )

            documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        **metadata,
                    },
                )
            )

        return documents
