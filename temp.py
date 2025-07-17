def forward(self, x, x_mark=None):
        # x: [B, L, C]
        non_missing = (x != 0).float()          # 1 where real value, 0 where placeholder
        x = x * non_missing                     # cancel contribution of the dummies

        # ---- value embedding ----
        v = self.value_embedding(x)             # [B, L, d_model]

        # renormalise so that “small” timesteps keep the same scale
        denom = non_missing.sum(-1, keepdim=True).clamp(min=1)      # [B, L, 1]
        v = v * self.c_in / denom              # scale back up to C‑independent magnitude

        # ---- add time / position embeddings ----
        if x_mark is None:
            out = v + self.position_embedding(v)
        else:
            out = v + self.position_embedding(v) + self.temporal_embedding(x_mark)

        return self.dropout(out)   
