import shutil
import unittest
from pathlib import Path
from unittest import TestCase

import PyPDF2

from mitools.exceptions import ArgumentTypeError, ArgumentValueError
from mitools.files import (
    can_move_file_or_folder,
    extract_pdf_metadata,
    extract_pdf_title,
    folder_in_subtree,
    folder_is_subfolder,
    handle_duplicated_filenames,
    remove_characters_from_filename,
    remove_characters_from_string,
    rename_file,
    rename_files_in_folder,
    rename_folders_in_folder,
    set_folder_pdfs_titles_as_filenames,
    set_pdf_title_as_filename,
)


class TestFolderIsSubfolder(TestCase):
    def setUp(self):
        self.root_folder = Path("/home/user/documents")
        self.sub_folder = Path("/home/user/documents/reports")
        self.non_sub_folder = Path("/home/user/pictures")

    def test_valid_subfolder(self):
        result = folder_is_subfolder(self.root_folder, self.sub_folder)
        self.assertTrue(result)

    def test_non_subfolder(self):
        result = folder_is_subfolder(self.root_folder, self.non_sub_folder)
        self.assertFalse(result)

    def test_same_folder(self):
        result = folder_is_subfolder(self.root_folder, self.root_folder)
        self.assertFalse(result)

    def test_non_existent_paths(self):
        result = folder_is_subfolder(
            Path("/non_existent_folder"), Path("/non_existent_folder/subfolder")
        )
        self.assertTrue(result)

    def test_string_input(self):
        result = folder_is_subfolder("/home/user", "/home/user/documents")
        self.assertTrue(result)


class TestFolderInSubtree(TestCase):
    def setUp(self):
        self.root = Path("/home/user/documents")
        self.branch = Path("/home/user/documents/reports")
        self.subfolder = Path("/home/user/documents/reports/2024")
        self.outside_folder = Path("/home/user/pictures")

    def test_folder_in_subtree_found(self):
        result = folder_in_subtree(self.root, self.subfolder, [self.branch])
        self.assertEqual(
            result, Path("/home/user/documents/reports").resolve(strict=False)
        )

    def test_no_folder_in_subtree(self):
        result = folder_in_subtree(self.root, self.subfolder, [self.outside_folder])
        self.assertIsNone(result)

    def test_branch_not_in_root_subtree(self):
        result = folder_in_subtree(
            self.root, self.outside_folder, [self.outside_folder]
        )
        self.assertIsNone(result)

    def test_same_folder_as_root(self):
        result = folder_in_subtree(self.root, self.root, [self.root])
        self.assertIsNone(result)  # Same folder should not be considered a subfolder

    def test_string_input(self):
        result = folder_in_subtree(
            str(self.root), str(self.subfolder), [str(self.branch)]
        )
        self.assertEqual(
            result, Path("/home/user/documents/reports").resolve(strict=False)
        )


class TestCanMoveFileOrFolder(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        self.source_file = self.test_dir / "source.txt"
        self.source_file.write_text("This is a test file.")
        self.source_folder = self.test_dir / "source_folder"
        self.source_folder.mkdir()
        self.destination_file = self.test_dir / "destination.txt"
        self.destination_folder = self.test_dir / "destination_folder"
        self.destination_folder.mkdir()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_valid_file_to_file_move(self):
        destination = self.test_dir / "new_name.txt"
        self.assertTrue(can_move_file_or_folder(self.source_file, destination))

    def test_file_to_existing_file_without_overwrite(self):
        self.destination_file.touch()  # Create destination file
        with self.assertRaises(ArgumentValueError):
            can_move_file_or_folder(self.source_file, self.destination_file)

    def test_file_to_existing_file_with_overwrite(self):
        self.destination_file.touch()  # Create destination file
        self.assertTrue(
            can_move_file_or_folder(
                self.source_file, self.destination_file, overwrite=True
            )
        )

    def test_directory_to_existing_file(self):
        self.assertFalse(
            can_move_file_or_folder(self.source_folder, self.destination_file)
        )

    def test_file_to_directory(self):
        destination = self.destination_folder / "new_file.txt"
        self.assertTrue(can_move_file_or_folder(self.source_file, destination))

    def test_directory_to_directory(self):
        new_directory = self.test_dir / "new_folder"
        new_directory.mkdir(exist_ok=True)
        self.assertTrue(
            can_move_file_or_folder(self.source_folder, new_directory, overwrite=True)
        )

    def test_source_not_exist(self):
        non_existent_file = self.test_dir / "non_existent.txt"
        with self.assertRaises(ArgumentValueError):
            can_move_file_or_folder(non_existent_file, self.destination_file)

    def test_destination_parent_not_exist(self):
        invalid_destination = Path("./non_existent_folder/destination.txt")
        with self.assertRaises(ArgumentValueError):
            can_move_file_or_folder(self.source_file, invalid_destination)

    def test_permission_denied(self):
        self.source_file.chmod(0o000)
        with self.assertRaises(PermissionError):
            can_move_file_or_folder(self.source_file, self.destination_file)
        self.source_file.chmod(0o644)

    def test_path_length_exceeded(self):
        long_name = "a" * 256 + ".txt"
        long_path = self.test_dir / long_name
        with self.assertRaises(OSError):
            can_move_file_or_folder(self.source_file, long_path)

    def test_insufficient_space(self):
        if self.source_file.stat().st_dev != self.destination_folder.stat().st_dev:
            with self.assertRaises(OSError):
                can_move_file_or_folder(self.source_file, self.destination_file)


class TestRenameFoldersInFolder(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        (self.test_dir / "folder 1").mkdir()
        (self.test_dir / "folder 2").mkdir()
        (self.test_dir / "already_exists").mkdir()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_default_rename(self):
        rename_folders_in_folder(self.test_dir)
        self.assertTrue((self.test_dir / "folder_1").exists())
        self.assertTrue((self.test_dir / "folder_2").exists())

    def test_attempt_mode(self):
        rename_folders_in_folder(self.test_dir, attempt=True)
        self.assertTrue((self.test_dir / "folder 1").exists())
        self.assertTrue((self.test_dir / "folder 2").exists())

    def test_custom_char_replacement(self):
        rename_folders_in_folder(
            self.test_dir, char_replacement=lambda name: name.replace(" ", "-")
        )
        self.assertTrue((self.test_dir / "folder-1").exists())
        self.assertTrue((self.test_dir / "folder-2").exists())

    def test_existing_target_folder(self):
        (self.test_dir / "folder_1").mkdir()  # Create a conflicting folder
        rename_folders_in_folder(self.test_dir)
        self.assertTrue((self.test_dir / "folder 1").exists())  # Original remains

    def test_no_change_for_identical_name(self):
        (self.test_dir / "folder_1").mkdir()
        rename_folders_in_folder(
            self.test_dir,
            char_replacement=lambda name: name,  # No change
        )
        self.assertTrue((self.test_dir / "folder_1").exists())

    def test_invalid_directory(self):
        with self.assertRaises(ValueError):
            rename_folders_in_folder("./non_existent_folder")


class TestRemoveCharactersFromString(TestCase):
    def test_default_removal(self):
        input_str = "file:name?with|invalid<characters>"
        expected = "filenamewithinvalidcharacters"
        result = remove_characters_from_string(input_str)
        self.assertEqual(result, expected)

    def test_path_object_input(self):
        path_input = Path("invalid/file:name.txt")
        expected = "invalidfilename.txt"
        result = remove_characters_from_string(str(path_input))
        self.assertEqual(result, expected)

    def test_custom_character_removal(self):
        input_str = "remove these characters!"
        characters_to_remove = r"[!]"
        expected = "remove these characters"
        result = remove_characters_from_string(input_str, characters_to_remove)
        self.assertEqual(result, expected)

    def test_no_characters_to_remove(self):
        input_str = "this string stays unchanged"
        result = remove_characters_from_string(input_str, "")
        self.assertEqual(result, input_str)

    def test_empty_string(self):
        input_str = ""
        result = remove_characters_from_string(input_str)
        self.assertEqual(result, "")

    def test_all_characters_removed(self):
        input_str = "removeall"
        characters_to_remove = r"[a-z]"
        result = remove_characters_from_string(input_str, characters_to_remove)
        self.assertEqual(result, "")

    def test_special_characters_in_input(self):
        input_str = "@hello$world!"
        expected = "@hello$world!"
        result = remove_characters_from_string(input_str, r'[\\/*?:"<>|]')
        self.assertEqual(result, expected)

    def test_non_matching_characters(self):
        input_str = "no_match_here"
        characters_to_remove = r"[XYZ]"
        result = remove_characters_from_string(input_str, characters_to_remove)
        self.assertEqual(result, input_str)

    def test_unicode_characters(self):
        input_str = "hello✨world"
        result = remove_characters_from_string(input_str, r"[✨]")
        self.assertEqual(result, "helloworld")


class TestRemoveCharactersFromFilename(TestCase):
    def setUp(self):
        self.test_file = Path("invalid:file?name.txt")
        self.expected_file = Path("invalidfilename.txt")

    def test_default_character_removal(self):
        result = remove_characters_from_filename(self.test_file)
        self.assertEqual(result.name, self.expected_file.name)

    def test_custom_character_removal(self):
        file_path = Path("file-name-to-remove-dash.txt")
        expected = Path("filenametoremovedash.txt")
        result = remove_characters_from_filename(file_path, characters=r"[-]")
        self.assertEqual(result.name, expected.name)

    def test_filename_without_illegal_characters(self):
        file_path = Path("valid_filename.txt")
        result = remove_characters_from_filename(file_path)
        self.assertEqual(result.name, "valid_filename.txt")

    def test_empty_filename(self):
        file_path = Path(".txt")  # No name, only extension
        result = remove_characters_from_filename(file_path)
        self.assertEqual(result.name, ".txt")

    def test_special_unicode_characters(self):
        file_path = Path("hello✨world.txt")
        expected = Path("helloworld.txt")
        result = remove_characters_from_filename(file_path, characters=r"[✨]")
        self.assertEqual(result.name, expected.name)

    def test_path_object_with_nested_folders(self):
        file_path = Path("some/folder/with/invalid:file?name.txt")
        expected = Path("some/folder/with/invalidfilename.txt")
        result = remove_characters_from_filename(file_path)
        self.assertEqual(result.name, expected.name)
        self.assertEqual(result.parent, Path("some/folder/with"))

    def test_filename_with_spaces(self):
        file_path = Path("file with spaces.txt")
        expected = Path("filewithspaces.txt")
        result = remove_characters_from_filename(file_path, characters=r"[ ]")
        self.assertEqual(result.name, expected.name)

    def test_non_existing_file_path(self):
        file_path = Path("non_existing:file.txt")
        expected = Path("non_existingfile.txt")
        result = remove_characters_from_filename(file_path)
        self.assertEqual(result.name, expected.name)


class TestHandleDuplicatedFilenames(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        self.file_path = self.test_dir / "test_file.txt"
        self.file_path.touch()  # Create the first version of the file

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_no_conflict(self):
        new_file_path = self.test_dir / "unique_file.txt"
        result = handle_duplicated_filenames(new_file_path)
        self.assertEqual(result, new_file_path)

    def test_single_conflict(self):
        conflicting_file = self.test_dir / "test_file.txt"
        conflicting_file.touch()

        result = handle_duplicated_filenames(conflicting_file)
        expected = self.test_dir / "test_file_1.txt"
        self.assertEqual(result, expected)

    def test_multiple_conflicts(self):
        for i in range(1, 3):
            (self.test_dir / f"test_file_{i}.txt").touch()

        result = handle_duplicated_filenames(self.file_path)
        expected = self.test_dir / "test_file_3.txt"
        self.assertEqual(result, expected)

    def test_no_file_extension(self):
        file_without_ext = self.test_dir / "test_file"
        file_without_ext.touch()  # Create the conflicting file

        result = handle_duplicated_filenames(file_without_ext)
        expected = self.test_dir / "test_file_1"
        self.assertEqual(result, expected)

    def test_non_existing_directory(self):
        non_existing_file = Path("./non_existing_folder/test_file.txt")
        result = handle_duplicated_filenames(non_existing_file)
        self.assertEqual(result, non_existing_file)


class TestRenameFile(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        self.test_file = self.test_dir / "invalid:file?name.txt"
        self.test_file.touch()  # Create the initial test file

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_default_rename_with_sanitization(self):
        rename_file(self.test_file)
        expected_file = self.test_dir / "invalidfilename.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_custom_rename(self):
        new_name = "custom_name.txt"
        rename_file(self.test_file, new_name=new_name)
        expected_file = self.test_dir / new_name
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_conflict_handling(self):
        conflict_file = self.test_dir / "invalidfilename.txt"
        conflict_file.touch()

        rename_file(self.test_file)
        expected_file = self.test_dir / "invalidfilename_1.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_rename_no_extension(self):
        no_ext_file = self.test_dir / "invalid:file?name"
        no_ext_file.touch()

        rename_file(no_ext_file)
        expected_file = self.test_dir / "invalidfilename"
        self.assertTrue(expected_file.exists())
        self.assertFalse(no_ext_file.exists())

    def test_non_existent_file(self):
        non_existent_file = self.test_dir / "non_existent.txt"
        with self.assertRaises(ArgumentValueError):
            rename_file(non_existent_file)

    def test_custom_name_with_conflict(self):
        conflict_file = self.test_dir / "custom_name.txt"
        conflict_file.touch()
        rename_file(self.test_file, new_name="custom_name.txt")
        expected_file = self.test_dir / "custom_name_1.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_custom_name_with_conflict_and_overwrite(self):
        conflict_file = self.test_dir / "custom_name.txt"
        conflict_file.touch()
        rename_file(self.test_file, new_name="custom_name.txt", overwrite=True)
        expected_file = self.test_dir / "custom_name.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())

    def test_callable_new_name(self):
        rename_file(self.test_file, new_name=lambda file: "custom_name.txt")
        expected_file = self.test_dir / "custom_name.txt"
        self.assertTrue(expected_file.exists())
        self.assertFalse(self.test_file.exists())


class TestRenameFilesInFolder(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        (self.test_dir / "file%&1.txt").touch()
        (self.test_dir / "file%&2.pdf").touch()
        (self.test_dir / "file%&3.TXT").touch()
        (
            self.test_dir / "non_file"
        ).mkdir()  # Create a folder to ensure non-files are skipped

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_default_rename_all_files(self):
        rename_files_in_folder(self.test_dir)
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"file1.txt", "file2.pdf", "file3.TXT"}
        self.assertEqual(renamed_files, expected_files)

    def test_rename_with_file_type_filter(self):
        rename_files_in_folder(self.test_dir, file_types=[".txt"])
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"file1.txt", "file3.TXT", "file%&2.pdf"}  # .pdf should remain
        self.assertEqual(renamed_files, expected_files)

    def test_rename_with_custom_function(self):
        def custom_renamer(file: str) -> str:
            return file.replace("file%&", "renamed")

        rename_files_in_folder(self.test_dir, renaming_function=custom_renamer)
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"renamed1.txt", "renamed2.pdf", "renamed3.TXT"}
        self.assertEqual(renamed_files, expected_files)

    def test_rename_with_duplicated_name(self):
        def custom_renamer(file: str) -> str:
            return file.replace("file%&", "file%&")  # Do nothing

        rename_files_in_folder(self.test_dir, renaming_function=custom_renamer)
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"file%&1_1.txt", "file%&2_1.pdf", "file%&3_1.TXT"}
        self.assertEqual(renamed_files, expected_files)

    def test_rename_with_duplicated_name_and_overwrite(self):
        def custom_renamer(file: str) -> str:
            return file.replace("file%&", "file%&")  # Do nothing

        rename_files_in_folder(
            self.test_dir, renaming_function=custom_renamer, overwrite=True
        )
        renamed_files = {f.name for f in self.test_dir.iterdir() if f.is_file()}
        expected_files = {"file%&1.txt", "file%&2.pdf", "file%&3.TXT"}
        self.assertEqual(renamed_files, expected_files)

    def test_rename_skips_non_files(self):
        rename_files_in_folder(self.test_dir)
        self.assertTrue((self.test_dir / "non_file").exists())

    def test_error_handling(self):
        def faulty_renamer(file: str) -> str:
            raise ValueError("Renaming failed")

        try:
            rename_files_in_folder(self.test_dir, renaming_function=faulty_renamer)
        except Exception as e:
            self.fail(f"rename_files_in_folder raised an unexpected exception: {e}")

    def test_rename_with_nonexistent_folder(self):
        with self.assertRaises(ArgumentValueError):
            rename_files_in_folder(Path("./non_existent_folder"))


class TestExtractPdfMetadata(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        self.pdf_with_metadata = self.test_dir / "with_metadata.pdf"
        with open(self.pdf_with_metadata, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata(
                {
                    "/Title": "Test PDF",
                    "/Author": "Jane Doe",
                    "/Subject": "Testing Metadata",
                }
            )
            writer.write(f)
        self.pdf_without_metadata = self.test_dir / "without_metadata.pdf"
        with open(self.pdf_without_metadata, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.write(f)
        self.corrupted_pdf = self.test_dir / "corrupted.pdf"
        with open(self.corrupted_pdf, "wb") as f:
            f.write(b"%PDF-1.4\nINVALID CONTENT")
        self.non_pdf_file = self.test_dir / "not_a_pdf.txt"
        with open(self.non_pdf_file, "w") as f:
            f.write("This is not a PDF file.")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_extract_metadata_valid_pdf(self):
        result = extract_pdf_metadata(self.pdf_with_metadata)
        expected = {
            "Title": "Test PDF",
            "Author": "Jane Doe",
            "Subject": "Testing Metadata",
            "Producer": "PyPDF2",
        }
        self.assertEqual(result, expected)

    def test_extract_metadata_empty_pdf(self):
        result = extract_pdf_metadata(self.pdf_without_metadata)
        self.assertEqual(result, {"Producer": "PyPDF2"})

    def test_non_existing_file(self):
        non_existing_file = self.test_dir / "non_existent.pdf"
        with self.assertRaises(ArgumentValueError):
            extract_pdf_metadata(non_existing_file)

    def test_corrupted_pdf(self):
        with self.assertRaises(ArgumentValueError):
            extract_pdf_metadata(self.corrupted_pdf)

    def test_non_pdf_file(self):
        with self.assertRaises(ArgumentValueError):
            extract_pdf_metadata(self.non_pdf_file)

    def test_large_pdf_with_metadata(self):
        large_pdf = self.test_dir / "large_with_metadata.pdf"
        with open(large_pdf, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Title": "Large Test PDF"})
            for _ in range(100):
                writer.add_blank_page(width=210, height=297)
            writer.write(f)
        result = extract_pdf_metadata(large_pdf)
        self.assertEqual(result, {"Title": "Large Test PDF", "Producer": "PyPDF2"})

    def test_unicode_metadata(self):
        unicode_pdf = self.test_dir / "unicode_metadata.pdf"
        with open(unicode_pdf, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Title": "Тестовый PDF"})  # Russian text
            writer.write(f)
        result = extract_pdf_metadata(unicode_pdf)
        self.assertEqual(result, {"Title": "Тестовый PDF", "Producer": "PyPDF2"})


class TestExtractPdfTitle(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        self.pdf_with_title = self.test_dir / "with_title.pdf"
        with open(self.pdf_with_title, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Title": "Test PDF Title"})
            writer.write(f)
        self.pdf_without_title = self.test_dir / "without_title.pdf"
        with open(self.pdf_without_title, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Author": "Jane Doe"})
            writer.write(f)
        self.corrupted_pdf = self.test_dir / "corrupted.pdf"
        with open(self.corrupted_pdf, "wb") as f:
            f.write(b"%PDF-1.4\nINVALID CONTENT")
        self.non_pdf_file = self.test_dir / "not_a_pdf.txt"
        with open(self.non_pdf_file, "w") as f:
            f.write("This is not a PDF file.")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_extract_title_from_valid_pdf(self):
        result = extract_pdf_title(self.pdf_with_title)
        self.assertEqual(result, "Test PDF Title")

    def test_extract_title_from_pdf_without_title(self):
        with self.assertRaises(ArgumentValueError) as context:
            extract_pdf_title(self.pdf_without_title)
        self.assertIn("has no title in its metadata", str(context.exception))

    def test_non_existent_file(self):
        non_existing_file = self.test_dir / "non_existent.pdf"
        with self.assertRaises(ArgumentValueError) as context:
            extract_pdf_title(non_existing_file)
        self.assertIn("is not a valid file path", str(context.exception))

    def test_extract_title_from_corrupted_pdf(self):
        with self.assertRaises(ArgumentValueError) as context:
            extract_pdf_title(self.corrupted_pdf)
        self.assertIn("Error reading PDF", str(context.exception))

    def test_extract_title_from_non_pdf_file(self):
        with self.assertRaises(ArgumentTypeError) as context:
            extract_pdf_title(self.non_pdf_file)
        self.assertIn("is not a valid PDF file", str(context.exception))


class TestSetPdfTitleAsFilename(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        self.pdf_with_title = self.test_dir / "with_title.pdf"
        with open(self.pdf_with_title, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Title": "Valid PDF Title"})
            writer.write(f)
        self.pdf_without_title = self.test_dir / "without_title.pdf"
        with open(self.pdf_without_title, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Author": "Jane Doe"})
            writer.write(f)
        self.non_pdf_file = self.test_dir / "not_a_pdf.txt"
        with open(self.non_pdf_file, "w") as f:
            f.write("This is not a PDF file.")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_valid_pdf_with_title(self):
        set_pdf_title_as_filename(self.pdf_with_title)
        expected_filename = self.test_dir / "Valid_PDF_Title.pdf"
        self.assertTrue(expected_filename.exists())

    def test_pdf_without_title(self):
        with self.assertRaises(ArgumentValueError) as context:
            set_pdf_title_as_filename(self.pdf_without_title)
        self.assertIn("has no title in its metadata", str(context.exception))

    def test_non_pdf_file(self):
        with self.assertRaises(ArgumentTypeError) as context:
            set_pdf_title_as_filename(self.non_pdf_file)
        self.assertIn("is not a valid PDF file", str(context.exception))

    def test_rename_with_duplicate_filename(self):
        duplicate_pdf = self.test_dir / "Valid_PDF_Title.pdf"
        duplicate_pdf.touch()  # Create a duplicate file
        set_pdf_title_as_filename(self.pdf_with_title)
        expected_filename = self.test_dir / "Valid_PDF_Title_1.pdf"
        self.assertTrue(expected_filename.exists())

    def test_overwrite_existing_file(self):
        duplicate_pdf = self.test_dir / "Valid_PDF_Title.pdf"
        duplicate_pdf.touch()  # Create a duplicate file
        set_pdf_title_as_filename(self.pdf_with_title, overwrite=True)
        self.assertTrue(duplicate_pdf.exists())
        self.assertFalse(self.pdf_with_title.exists())  # Original should be renamed


class TestSetFolderPDFsTitlesAsFilenames(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        self.pdf_with_title = self.test_dir / "with_title.pdf"
        self.create_pdf_with_metadata(self.pdf_with_title, title="Test Title")
        self.pdf_without_title = self.test_dir / "without_title.pdf"
        self.create_pdf_with_metadata(self.pdf_without_title, title=None)
        self.non_pdf_file = self.test_dir / "not_a_pdf.txt"
        self.non_pdf_file.write_text("This is a text file.")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @staticmethod
    def create_pdf_with_metadata(file_path: Path, title: str = None):
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_blank_page(width=72, height=72)  # Add a blank page
        if title:
            pdf_writer.add_metadata({"/Title": title})
        with open(file_path, "wb") as f:
            pdf_writer.write(f)

    def test_valid_folder_with_pdfs(self):
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=True)
        expected_file = self.test_dir / "Test_Title.pdf"
        self.assertTrue(expected_file.exists())

    def test_pdf_without_title(self):
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=True)

    def test_skip_non_pdf_files(self):
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=True)
        self.assertTrue(self.non_pdf_file.exists())  # Non-PDF file remains unchanged

    def test_invalid_folder_path(self):
        with self.assertRaises(ArgumentValueError):
            set_folder_pdfs_titles_as_filenames("./non_existent_folder")

    def test_attempt_mode(self):
        set_folder_pdfs_titles_as_filenames(self.test_dir, attempt=True)
        old_file = self.test_dir / "with_title.pdf"
        new_file = self.test_dir / "Test_Title.pdf"
        self.assertTrue(old_file.exists())  # Ensure no renaming occurred
        self.assertFalse(new_file.exists())

    def test_dont_overwrite_existing_file(self):
        duplicate_pdf = self.test_dir / "Test_Title.pdf"
        self.create_pdf_with_metadata(duplicate_pdf, title="Test Title")
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=False)
        self.assertTrue((self.test_dir / "Test_Title_1.pdf").exists())
        self.assertFalse((self.test_dir / "Test_Title.pdf").exists())

    def test_overwrite_existing_file(self):
        duplicate_pdf = self.test_dir / "Test_Title.pdf"
        self.create_pdf_with_metadata(duplicate_pdf, title="Test Title")
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=True)
        self.assertTrue((self.test_dir / "Test_Title.pdf").exists())
        self.assertFalse((self.test_dir / "Test_Title_1.pdf").exists())

    def test_failure_handling_in_pdf_processing(self):
        corrupted_pdf = self.test_dir / "corrupted.pdf"
        corrupted_pdf.write_text("This is not a valid PDF.")
        set_folder_pdfs_titles_as_filenames(self.test_dir)


if __name__ == "__main__":
    unittest.main()
