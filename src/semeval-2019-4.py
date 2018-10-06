from datatasks import parse_xml

def main():
    data = parse_xml.parse_text('../data/test/articles_sample.xml')
    print(data.head())

if __name__ == '__main__':
    main()