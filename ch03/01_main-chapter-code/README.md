# Chapter 3: Coding Attention Mechanisms

### Main Chapter Code

- [ch03.ipynb](ch03.ipynb) contains all the code as it appears in the chapter

### Optional Code

- [multihead-attention.ipynb](multihead-attention.ipynb) is a minimal notebook with the main data loading pipeline implemented in this chapter

### Explanation of the code

- Let me explain the line `keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)` in detail:
- Purpose:
    - This line reshapes the keys tensor to support multi-head attention by splitting the embedding dimension into multiple heads.
- Let's break it down step by step:
    - Original Shape:
        - Before this line, keys has shape [b, num_tokens, d_out]
        - Where:
            - b = batch size
            - num_tokens = number of tokens in sequence
            - d_out = total output dimension
    - New Shape:
        - After reshaping, keys has shape [b, num_tokens, num_heads, head_dim]
        - Where:
            - b = batch size (unchanged)
            - num_tokens = number of tokens (unchanged)
            - num_heads = number of attention heads
            - head_dim = dimension per head (d_out / num_heads)
- Example:
    - ```
        # Let's say we have:
            - b = 2          # batch size
            - num_tokens = 6 # sequence length
            - d_out = 8      # total output dimension
            - num_heads = 2  # number of attention heads
            - head_dim = 4   # dimension per head (8/2)

            # Original keys shape: [2, 6, 8]
            # After view: [2, 6, 2, 4]
    ```

- In multi head attention, for fatser parallel computation of all the head (queries, keys, values), we reshape the keys tensor to [b, num_tokens, num_heads, head_dim]. original dimension is [b, num_tokens, d_out]. d_out is num_heads * head_dim.
- **Benefits**:
    - Parallel processing of multiple attention patterns
    - Each head can learn different aspects of the input
    - More efficient computation than running separate attention mechanisms