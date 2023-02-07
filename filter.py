from dagshub.streaming import install_hooks
import csv
import numpy as np


with open('speech-accent-archive/speakers_all.csv') as list_all:
    data = list(csv.reader(list_all))
    data = np.array(data)
    header = data[0,:9]
    data = data[1:,:9]

    # count distinct native languages
    values, counts = np.unique(data[:,4], return_counts=True)
    print(f'There are {len(values)} distinct native languages in this data set.')
    min_count = 40
    important_lang = counts>=min_count

    # down sampling english as it has the most samples
    english_running = 0
    english_samples = np.sort(counts)[-2]  # using as many samples as the second most common language
    counts[values=='english'] = english_samples

    print(f'There are {np.sum(important_lang)} languages with at least {min_count} samples.')
    print(f'Those make up {sum(counts[important_lang])} out of {data.shape[0]} samples and a ration of {sum(counts[important_lang])/data.shape[0]} of the full data set.')

    # => use those 9 languages with at least 40 samples!
    data_use = data[np.isin(data[:,4],values[important_lang])]
    with open('speech-accent-archive/speakers_use.csv', 'w') as writefile:
        writer = csv.writer(writefile)
        writer.writerow(header)
        for row in data_use:
            if row[4] == 'english' and english_running >= english_samples:
                continue
            else:
                english_running += 1
            writer.writerow(row)
    with open('speech-accent-archive/overview.csv', 'w') as over_f:
        writer = csv.writer(over_f)
        writer.writerow(['native_language', 'count'])
        for i in range(np.sum(important_lang)):
            writer.writerow([values[important_lang][i],counts[important_lang][i]])
        writer.writerow(['rest', data.shape[0]-sum(counts[important_lang])])
