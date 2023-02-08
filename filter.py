from dagshub.streaming import DagsHubFilesystem
import csv
import numpy as np
import os


def filter_data(min_samples=40):
    """Determining what samples of the data set to use for classifying.

    Using only classes that have at least `min_samples`.
    Creates file 'data/classes.csv' with used classes and sample number and file 'data/speakers_use.csv' with the samples to use.
    As one class ('native_language'='english') has way more samples it is cut to the same size as the second biggest class.
    :param min_samples: Minimum amount of samples per class
    :return: number of samples used
    """
    # using dagshub file system to load data if not yet present in physical system
    if not os.path.exists('speech-accent-archive/speakers_all.csv'):
        os.chdir('speech-accent-archive/')
        fs = DagsHubFilesystem()
        with fs.open('speakers_all.csv'):
            pass
        os.chdir('../')

    # Open file with information about all samples
    with open('speech-accent-archive/speakers_all.csv') as list_all:
        data = list(csv.reader(list_all))
        data = np.array(data)
        # Discarding header and empty columns
        header = data[0,:9]
        data = data[1:,:9]

        # Count distinct native languages and their frequencies
        values, counts = np.unique(data[:,4], return_counts=True)
        print(f'There are {len(values)} distinct native languages in this data set.')
        # Use only classes with at least `min_samples`samples
        important_lang = (counts >= min_samples)

        # down sampling english as it has the most samples
        english_running = 0
        english_samples = np.sort(counts)[-2]  # using as many samples as the second most common language
        counts[values == 'english'] = english_samples

        print(f'There are {np.sum(important_lang)} languages with at least {min_samples} samples.')
        print(f'Those make up {sum(counts[important_lang])} out of {data.shape[0]} samples and a ration of {sum(counts[important_lang])/data.shape[0]} of the full data set.')

        # Use those 5 languages with at least 60 samples!
        data_use = data[np.isin(data[:,4],values[important_lang])]
        # save in csv file in same shape as before
        with open('data/speakers_use.csv', 'w') as writefile:
            writer = csv.writer(writefile)
            writer.writerow(header)
            for row in data_use:
                # cut english if too many
                if row[4] == 'english' and english_running >= english_samples:
                    continue
                else:
                    english_running += 1
                writer.writerow(row)
        # save overview over classes
        with open('data/classes.csv', 'w') as over_f:
            writer = csv.writer(over_f)
            writer.writerow(['native_language', 'count'])
            for i in range(np.sum(important_lang)):
                writer.writerow([values[important_lang][i],counts[important_lang][i]])
            writer.writerow(['rest', data.shape[0]-sum(counts[important_lang])])
    return sum(counts[important_lang])
