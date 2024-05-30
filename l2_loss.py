def l2_loss(arr_1 : In[Array[Array[Array[float]]]], arr_2 : In[Array[Array[Array[float]]]], height : In[int], width : In[int], color : In[int]) -> float:
    i : int = 0
    j : int = 0
    k : int = 0
    s : float = 0.0
    while (i < height, max_iter := 300):
        j = 0
        while (j < width, max_iter := 300):
            k = 0
            while (k < color, max_iter := 3):
                s = s + pow(arr_1[i][j][k] - arr_2[i][j][k], 2.0)
                k = k + 1
            j = j + 1
        i = i + 1
    return s / (height * width * 3)

d_l2_loss = rev_diff(l2_loss)
