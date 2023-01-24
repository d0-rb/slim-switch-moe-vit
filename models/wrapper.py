def forward_block_vanilla(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


def forward_block_w_attn(self, x):
    prev_attn = None
    if isinstance(x, (tuple)):
        x, prev_attn = x
    x = self._forward(self, x)
    attn = self.attn.x_cls_attn.sum(dim=1)
    patch_attn = attn[:, 1::]
    if prev_attn is not None:
        patch_attn = (patch_attn + prev_attn) * 0.5
    return x, patch_attn


def forward_block_w_full_attn(self, x):
    prev_attn = None
    if isinstance(x, (tuple)):
        x, prev_attn = x
    x = self._forward(self, x)
    if self.is_last:
        return x

    attn = self.attn.attn.sum(dim=1)
    if prev_attn is not None:
        attn = (attn + prev_attn) * 0.5
    return x, attn


def forward_block_no_attn(self, input_):
    x, attn = self._forward(self, input_)
    return x
