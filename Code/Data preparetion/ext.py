import csv
import os
import json
import re
import string
from pprint import pprint


def get_data_by_field(book_data, field):
    """Retrieve the values of a specific MARC field from book data"""
    return [e[field] for e in book_data['fields'] if e.get(field) is not None]

def extract_title(book_data):
    """Extract the title from the MARC 245 field"""
    values = get_data_by_field(book_data, '245')
    assert len(values) == 1
    title = []
    c_end_char = None
    has = False
    for e in values[0]['subfields']:
        if 'c' in e:
            c_end_char = e['c'][-1]
            if c_end_char not in string.punctuation:
                c_end_char = None
            if '=' in e['c'] and '/' in e['c']:
                idx1 = e['c'].index('=')
                idx2 = e['c'].index('/')
                if -1 < idx1 < idx2:
                    if len(title) > 0 and title[-1].endswith('/'):
                        title[-1] = title[-1][0:-1].strip()
                    title.append("= " + (e['c'][idx1 + 1:idx2].strip()))
                    has = True
            continue
        sub_values = [se for se in e.values()]
        assert len(sub_values) == 1
        if c_end_char:
            title.append(c_end_char)
            c_end_char = None
            has = True
        title.append(sub_values[0].strip())
    title = ' '.join(title)
    title = title.strip()
    if title.endswith('/'):
        title = title[0:-1].strip()
    return title


def extract_abstract(data):
    """Extract abstract from field 520"""
    values = get_data_by_field(data, '520')
    abstract = []
    for value in values:
        ind1 = value['ind1'].strip()
        if not (ind1 == '' or ind1 == '3'):
            continue
        for e in value['subfields']:
            sub_values = [se for se in e.values()]
            assert len(sub_values) == 1
            abstract.append(sub_values[0].strip())
    return '\n'.join(abstract)

def lcc_standarize(lcc):
    """Standardize LCC codes into consistent format"""
    lcc_std = re.match(r'^([A-Za-z]+)\W*(\d+(\.\d+)?)', lcc)
    if lcc_std:
        lcc_std = lcc_std.group()
        lcc_std = lcc_std.upper().replace(" ", "")
    else:
        lcc_std = ''
    return lcc_std


def extract_lcc(data):
    """Extract LCC call numbers from fields 050 and 090"""
    values1 = get_data_by_field(data, '050')
    values2 = get_data_by_field(data, '090')
    values = values1 + values2
    lccs = []
    lccs_std = []
    for value in values:
        for e in value['subfields']:
            if 'a' in e:
                sub_values = [se for se in e.values()]
                assert len(sub_values) == 1
                lcc = sub_values[0].strip()
                lccs.append(lcc)
                lccs_std.append(lcc_standarize(lcc))
    return ' ; '.join(lccs), ' ; '.join(lccs_std)


def extract_table_of_contents(data):
    """Extract table of contents from field 505"""
    values = get_data_by_field(data, '505')
    tocs = []
    for value in values:
        for e in value['subfields']:
            sub_values = [se for se in e.values()]
            assert len(sub_values) == 1
            tocs.append(sub_values[0].strip())
    assert len(tocs) > 0
    return '\n'.join(tocs).strip()


def extract_publisher_year(data):
    """Extract publication year from fields 008 or 264"""
    values = get_data_by_field(data, '008')
    if len(values) > 0:
        return values[0][7:11].strip(), values[0][11:15].strip()
    else:
        values = get_data_by_field(data, '264')
        assert len(values) == 1
        if values[0]['ind2'] == '1':
            for e in values[0]['subfields']:
                if 'c' in e:
                    sub_values = [se for se in e.values()]
                    assert len(sub_values) == 1
                    year = re.search(r'\d+', sub_values[0]).group()
                    return year, ''
        return None, None


def extract_subject_headings(book_data):
    """Extract LCSH and FAST subject headings"""
    fields = ['650']
    subjects_lcsh = []
    subjects_fast = []

    for field in fields:
        values = get_data_by_field(book_data, field)
        for value in values:
            # Check for LCSH (indicator2 = 0)
            if value['ind2'].strip() == '0':
                subject = [e['a'].strip() for e in value['subfields'] if 'a' in e]
                subjects_lcsh.extend(subject)
            # Check for FAST (indicator2 = 7 and subfield '2' == 'fast')
            elif value['ind2'].strip() == '7':
                fast_indicator = [e['2'].strip().lower() for e in value['subfields'] if '2' in e]
                if fast_indicator and 'fast' in fast_indicator:
                    subject = [e['a'].strip() for e in value['subfields'] if 'a' in e]
                    subjects_fast.extend(subject)
                    print(f"Extracted FAST subject headings: {subject}")
    return '; '.join(subjects_lcsh), '; '.join(subjects_fast)


def extract_bibli(data_dir, save_file_path):
    """Main function to iterate over all JSON files and extract bibliographic data"""
    with open(save_file_path, mode='w', newline='', encoding='utf-8-sig') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['lcc', 'lcc_std', 'start_year', 'end_year', 'title', 'abstract', 'toc', 'lcsh_subject_headings', 'fast_subject_headings'])
        
        filenames = os.listdir(data_dir)
        filenames.sort()
        print(f"{len(filenames)} files to be processed.")

        count = 0
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(data_dir, filename)
            print(f"Processing file: {filename}")

            try:
                with open(file_path, mode='r', encoding='utf-8') as infile:
                    data = json.load(infile)
                    
                    # Check if data is a list (multiple records)
                    if isinstance(data, list):
                        for book_data in data:
                            process_and_write_book_data(writer, book_data)
                            count += 1
                    else:
                        # Single record
                        process_and_write_book_data(writer, data)
                        count += 1
                        
                    print(f"Successfully wrote row {count}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"Extraction complete. {count} records processed.")


def process_and_write_book_data(writer, book_data):
    """Process a single book entry and write it to CSV"""
    try:
        title = extract_title(book_data)
        abstract = extract_abstract(book_data)
        lcc, lcc_std = extract_lcc(book_data)
        toc = extract_table_of_contents(book_data)
        syear, eyear = extract_publisher_year(book_data)
        lcsh_subject_headings, fast_subject_headings = extract_subject_headings(book_data)
        
        writer.writerow([lcc, lcc_std, syear, eyear, title, abstract, toc, lcsh_subject_headings, fast_subject_headings])
    except Exception as e:
        print(f"Error processing a book record: {e}")


if __name__=='__main__':
    data_dir = '/mnt/llm4cat/data-20241004/eng-summary-and-toc'
    save_file_path = '/home/jl1609@students.ad.unt.edu/bibli2.csv'
    extract_bibli(data_dir, save_file_path)
    print("ok")
