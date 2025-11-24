####################
# CNN Architecture #
####################
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with:
        - 2 conv layers + skip connection."""

    def __init__(self, channels): # channels: depth of data (ie. # of filters)
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels) # normalization for better learning
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class CNN(nn.Module):
    """
    AlphaZero-style architecture:
    - CNN with skip connections
    - Policy head: move winning probabilities (7 columns)
    - Value head: position winning advantage ([-1 = opponent advantage, 1 = current player advantage])
    We will refer to the notation in PUCT formula: https://substackcdn.com/image/fetch/$s_!8aNs!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdbf3bcbe-836f-49b3-93d4-af0c48219f24_2212x1437.png
    """

    def __init__(self, num_res_blocks=5, channels=128):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.channels = channels

        # Input layer (2D for each player's board pieces)
        self.conv_input = nn.Conv2d(2, channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)])

        # Policy head: P(s,a) move probabilities (s = state, a = action)
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32*6*7, 7)

        # Value head: V(s) position evaluation
        self.value_conv = nn.Conv2d(channels, 16, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16*6*7, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input layer
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy head: P(s,a) move probabilities
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head: V(s) position evaluation
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # get activation range [-1, 1]

        return p, v

    def predict(self, state, valid_moves):
        """Predict policy P(s,a) and V(s) for current state."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                state_t = state_t.cuda()

            raw, value = self(state_t) # raw output: (1, 7), value: (1, 1)
            invalid_filter = torch.full((7,), float('-inf')) # Initialize valid moves with -inf

            # Set valid moves to 0
            invalid_filter[valid_moves] = 0

            if next(self.parameters()).is_cuda:
                invalid_filter = invalid_filter.cuda()

            # Valid moves: raw output + 0 = raw output
            # Invalid moves: raw output + (-inf) = -inf
            raw_1d = raw[0]
            valid_out = raw_1d + invalid_filter
            policy = F.softmax(valid_out, dim=0).cpu().numpy() # (-inf) -> 0

            return policy, value.item()