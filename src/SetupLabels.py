def main():
    '''
    Processes text files containing statements of intent (SOI) 
    and letters of recommendation (LOR) from a source directory, 
    copies those with more than 10 characters to a
    target directory, cleans their content, and stores the resulting text,
    along with the corresponding student IDs and submission filenames, in a
    pandas DataFrame that is saved to a CSV file.

    The script reads the content of the SOI files, removes certain lines of
    text based on predefined rules, fixes some formatting issues, replaces
    certain deceptive characters, and writes the cleaned content to new files
    with the same name as the originals but with the suffix "_modified.txt".

    The script also generates a pandas DataFrame with three columns: "ID",
    "Files", and "Text". The "ID" column contains the student IDs extracted
    from the filenames of the processed SOI files, the "Files" column contains
    the submission filenames without the extension, and the "Text" column
    contains the cleaned content of the corresponding files.

    The DataFrame is saved to a CSV file named "df_real.csv" in the same
    directory as the script. The file can be read using the pandas function
    `pd.read_csv("df_real.csv", dtype=str)` with dtype=str to preserve any
    leading zeros in the "ID" column.

    The script could be extended to generate new observations based on each
    real one, by prompting users to generate new statements of intent based on
    the cleaned content in the "Text" column.
    '''

    import os 
    from pathlib import Path
    import shutil
    import pandas as pd
    import re

    src_dir = "ChatGPTProjectData/Redacted_corrected"
    target_dir = "TrueSOI"
    save_folder = Path(target_dir)

    save_folder.mkdir(exist_ok=True)

    for root, _, files in os.walk(src_dir):
        #SOI
        if os.path.basename(root) == "SOI":
            for file in files:
                if file.endswith('.txt'):
                    # construct the source and target file paths
                    src_file_path = os.path.join(root, file)
                    target_file_path = os.path.join(target_dir, file)
                    # check the number of characters in the file to only copy the ones with relevant content (more than 10 characters)
                    with open(src_file_path, "r") as f:
                        n_char = len(f.read())
                    if n_char > 10:
                    # copy the file to the target directory if the file contains more than 10 characters
                        shutil.copy(src_file_path, target_file_path)
        
        # LOR
        if os.path.basename(root) == "LOR":
            for file in files:
                if file.endswith('.txt'):
                    # construct the source and target file paths
                    src_file_path = os.path.join(root, file)
                    target_file_path = os.path.join(target_dir, file)
                    # check the number of characters in the file to only copy the ones with relevant content (more than 10 characters)
                    with open(src_file_path, "r") as opened_file:
                        text = opened_file.read()

                        # Regular expression pattern to match the section: Only consider from \nsignature up to HEAD SECTION
                        pattern = r"(\n)+(signature)[\s\S]*(HEADER SECTION)"
                        match = re.search(pattern, text)
                        # If match pattern and have more than 50 words
                        if match and len(match.group(0).strip().split()) > 50:
                            extracted_text = match.group(0)
                            extracted_text = "\n".join([line for line in extracted_text.split("\n") if len(line.strip().split()) > 7])
                            extracted_text = [line for line in extracted_text.split("\n") if "first name" not in line or "last name" not in line or "recommender information" not in line]
                            extracted_text = "\n".join(extracted_text).strip()
                            
                            with open(target_file_path, "w") as f:
                                f.write(extracted_text.replace("HEADER SECTION","").replace("signature\n\n","").replace("\x0c","").lstrip().strip())


    # Create empty lists to store information
    IDs = []
    SubmissionFile = []
    TypeDoc = []
    files_content = []
    # If a line have any of these strings, then is ignored 
    rules_to_ignore_line = ['subject:', '|page', "statement of purpose for","gsas ms data analytics", "please describe your purpose", "include an explanation of research and study interests, indicating how they relate","undergraduate program, relevant work experience and", "441 e. fordham road"]

    # Clean new txt files
    for file in os.listdir(target_dir):
        # Only process .txt files and the ones without the "_modified" string, since there could be altered versions. In case we find altered versions, we want to overwrite them.
        if file.endswith(".txt") and "_modified" not in file:
            # Register and append Student ID, File_Labels, and Cleaned content
            file_labels = file.split("_")
            IDs.append(file_labels[0]); SubmissionFile.append(file_labels[1]); TypeDoc.append("SOI" if "SOI" in file else "LOR") # Example of filename: 0002306584_041828950_2022Spring_MSCS_MSCS_SOI1_redacted.txt
            # Open the file and clean its content
            with open(os.path.join(target_dir, file), 'r') as opened_file, open(os.path.join(target_dir, f'{file.split(".")[0]}_modified.txt'), 'w') as output_file:
                for line in opened_file:
                    # Ignore certain lines of text
                    if not any(to_exclude in line for to_exclude in rules_to_ignore_line) and line.strip() not in ("statement of purpose", "statement of intent", "REDACTED REDACTED", "R"):
                        # Fix formatting issues. The original files have a misleading structure where there were lines added in places not supposed.
                        if not line.endswith(".\n"):
                            line = " " + line.strip()
                        # With the formatting structure fixed, we replace some deceptive characters.
                        line = line.replace("\x0c","").replace("REDACTED REDACTED", "REDACTED").replace("REDACTEDREDACTED", "REDACTED").replace("  ", " ").replace("\n\n", "\n")
                        # Write the cleaned content to the output file
                        output_file.write(line)

            with open(os.path.join(target_dir, f'{file.split(".")[0]}_modified.txt'), 'r') as content:
                # Add cleaned content
                files_content.append(content.read().strip().replace("REDACTED REDACTED","REDACTED").replace("REDACTEDREDACTED","REDACTED"))

    # Create a pandas DataFrame with the information
    IDS_Series = pd.Series(IDs,name="ID")
    Files_Series = pd.Series(SubmissionFile, name="Files")
    TypeDoc_Series = pd.Series(TypeDoc, name="TypeDoc")
    files_content_Series = pd.Series(files_content, name="Text")

    df = pd.DataFrame([IDS_Series, Files_Series, TypeDoc_Series, files_content_Series]).T
    # Store our dataframe into a .csv file.
    df.to_csv("df_real.csv",index=False)

if __name__ == "__main__":
    main()
