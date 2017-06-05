def angry_not_angry_labels(labels):
    new_labels = []
    for i in range(len(labels)):
        if labels[i] == 0:
            new_labels.append(1)
        else:
            new_labels.append(0)

    return new_labels
