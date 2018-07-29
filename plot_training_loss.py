import matplotlib.pyplot as plt


def get_loss(file_name):
    with open(file_name, 'r') as fp:
        for line in fp:
            if 'loss=' not in line:
                continue
            yield float(line.split('loss=')[1].strip())


def main():
    log_file = './training.log'
    x = []
    y = []
    for ix, loss in enumerate(get_loss(log_file)):
        x.append(ix)
        y.append(loss)
    plt.plot(x, y)
    plt.title('Training loss over time')
    plt.ylabel('Loss')
    plt.xlabel('Batches')
    plt.savefig('training_loss.png')


if __name__ == '__main__':
    main()
