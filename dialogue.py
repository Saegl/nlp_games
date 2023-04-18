import torch
from torch import nn, optim
import json
from tqdm.auto import tqdm, trange


MESSAGE_MARKER = ">"  # start or end of message


def load_dialogues(fname: str) -> list[str]:
    with open(fname, encoding="utf8") as f:
        data = list(json.load(f).values())

    output = []
    for entry in data:
        dialogue = (
            "".join([MESSAGE_MARKER + t for t in entry["turns"]]) + MESSAGE_MARKER
        )
        output.append(dialogue)
    return output


train_dialogues = load_dialogues("dialogues/data/train.json")

print("TRAIN DIALOGUES COUNT:", len(train_dialogues))
print("example:", train_dialogues[0])


OOA = "\0"  # out of alphabet

alphabet = set(char for dialogue in train_dialogues for char in dialogue)
alphabet.add(OOA)

index_to_char = sorted(alphabet)
char_to_index = {v: k for k, v in enumerate(index_to_char)}

OOA_INDEX = char_to_index[OOA]


print("Alphabet:", "".join(index_to_char))
print("Alphabet size:", len(alphabet))

assert "a" == index_to_char[char_to_index["a"]]

eye = torch.eye(len(alphabet))


class CharGenRNN(nn.Module):
    def __init__(self, alphabet_size, hidden_size) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=alphabet_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.o2o = nn.Linear(hidden_size, alphabet_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor, h0=None):
        if h0 is None:
            rnn_out, hidden = self.rnn(input_tensor)  # shape: (seq_len, hidden_size)
        else:
            rnn_out, hidden = self.rnn(input_tensor, h0)
        lin_out = self.o2o(rnn_out)  # shape (seq_len, alphabet_size)
        log_probs = self.softmax(lin_out)  # shape (seq_len, )
        return log_probs, hidden


def to_input_tensor(s: str) -> torch.Tensor:
    indicies = [char_to_index[c] for c in s]
    return torch.stack([eye[i] for i in indicies])


def append_context(
    model: CharGenRNN, input_tensor: torch.Tensor, old_context: torch.Tensor
) -> torch.Tensor:
    _, hidden = model(input_tensor, old_context)
    return hidden


model = torch.load("dialogue.model").to("cpu")
hidden = torch.zeros(1, 250)
MAX_ANS_LEN = 250
last_letter = ""

with torch.no_grad():
    while True:
        prefix = ">" if last_letter != ">" else ""
        question = input("Q> ")
        input_tensor = to_input_tensor(prefix + question + MESSAGE_MARKER)
        hidden = append_context(model, input_tensor, hidden)

        letter = None
        ans_length = 0
        for _ in range(MAX_ANS_LEN):
            output, hidden = model(input_tensor, hidden)

            topv, topi = output.topk(1)

            char_index = topi[0][0]
            letter = index_to_char[char_index]
            print(letter, end="")
            last_letter = letter
            if letter == MESSAGE_MARKER:
                break
            input_tensor = to_input_tensor(letter)
        print()
