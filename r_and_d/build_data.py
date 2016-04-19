# encoding: utf-8
"""
Extract and nicely save the JSON test data
"""
import json
import codecs

def clean_up_json():
    test_data = u'mnras_2015.json'
    with codecs.open(test_data, 'r', 'utf-8') as f:
        data = json.load(f)

    for entry in data['response']['docs']:
        if entry.get(u'keyword', None) is None: 
            continue

        # Make the fulltext file (x-value)
        filename = u'data/{}.txt'.format(entry['bibcode'])
        with codecs.open(filename, 'w', 'utf-8') as f:
            f.write(entry['body'])

        # Make the keyword file (y-value)
        filename = u'data/{}.key'.format(entry['bibcode'])
        with codecs.open(filename, 'w', 'utf-8') as f:

            keywords = []
            for keyword in entry['keyword']:
                k = keyword.replace('-', ':').split(':')
                keywords.extend(k)

            keywords = '\n'.join(keywords)
            f.write(u'{}'.format(keywords))


if __name__ == '__main__':
    clean_up_json()
 
