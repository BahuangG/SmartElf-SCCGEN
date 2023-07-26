import statistics
from nltk.translate import bleu_score
from tqdm import tqdm

chencherry = bleu_score.SmoothingFunction()

# Evaluate perfect prediction & BLEU score of our approach
for k in [1]:

    print('k candidates: ', k)
    path_targets = 'newdata/nl.csv'
    path_predictions = 'newdata/sml-1.csv'

    tgt = [line.strip() for line in open(path_targets)]
    pred = [line.strip() for line in open(path_predictions)]
    print(len(tgt))
    print(len(pred))


    count_perfect = 0
    BLEUscore = []
    for i in tqdm(range(len(tgt))):
        best_BLEU = 0
        target = tgt[i]
        for prediction in pred[i * k:i * k + k]:
            if " ".join(prediction.split()) == " ".join(target.split()):
                count_perfect += 1
                best_BLEU = bleu_score.sentence_bleu([target], prediction, smoothing_function=chencherry.method1)
                break
            current_BLEU = bleu_score.sentence_bleu([target], prediction, smoothing_function=chencherry.method1)
            if current_BLEU > best_BLEU:
                best_BLEU = current_BLEU
        BLEUscore.append(best_BLEU)

    print(f'PP    : %d/%d (%s%.2f)' % (count_perfect, len(tgt), '%', (count_perfect * 100) / len(tgt)))
    print(f'BLEU mean              : ', statistics.mean(BLEUscore))

    # with open('newdata/bleu1.txt', 'w') as fs:
    #     for bleu in BLEUscore:
    #         fs.write(str(bleu) + '\n')
