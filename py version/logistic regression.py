{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature: 0 , Score: 0.16320\n",
      "Feature: 1 , Score: -0.64301\n",
      "Feature: 2 , Score: 0.48497\n",
      "Feature: 3 , Score: -0.46190\n",
      "Feature: 4 , Score: 0.18432\n",
      "Feature: 5 , Score: -0.11978\n",
      "Feature: 6 , Score: -0.40602\n",
      "Feature: 7 , Score: 0.03772\n",
      "Feature: 8 , Score: -0.51785\n",
      "Feature: 9 , Score: 0.26540\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAi8AAAGdCAYAAADaPpOnAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAchElEQVR4nO3df5CV9X3/7+cuhAUTFiQCK7p01XYCBAPKBorW2JatoNTWGWs1JRUpxc7UTdS1TiFtxJSaxYwa6o9qtdpOpzLaNmNqTcsMhRqq3QhC6EQr2CalUpgFGSqL0CKy+/0jk813PwKC8XB4L9c1c/+x97nvc784w8w+5r33Oaemp6enJwAAhait9gAAAMdDvAAARREvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFCUgdUe4MPW3d2d7du3Z+jQoampqan2OADAMejp6cnevXszZsyY1NYefW2l38XL9u3b09jYWO0xAIAPYOvWrTn77LOPeky/i5ehQ4cm+cE/vr6+vsrTAADHoqurK42Njb2/x4+m38XLD/9UVF9fL14AoDDHcsuHG3YBgKKIFwCgKOIFACiKeAEAiiJeAICiiBcAoCjiBQAoingBAIoiXgCAoogXAKAo4gUAKIp4AQCKIl4AgKL0u2+Vpn9pWvjNao/wHluWzq72CACnNCsvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARREvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFFOSLw89NBDaWpqyuDBgzNt2rSsXbv2mM576qmnUlNTk6uuuqqyAwIAxah4vDz99NNpa2vL4sWLs2HDhkyaNCkzZ87Mzp07j3reli1b8ju/8zu55JJLKj0iAFCQisfLfffdlwULFmTevHmZMGFCHnnkkZx22ml54oknjnjOoUOHMmfOnHz5y1/OueeeW+kRAYCCVDRe3nnnnaxfvz4tLS0/umBtbVpaWtLR0XHE8/7gD/4go0aNyvz589/3GgcOHEhXV1efDQDovyoaL7t27cqhQ4cyevToPvtHjx6dzs7Ow57zwgsv5PHHH89jjz12TNdob2/PsGHDerfGxsYfe24A4OR1Ur3baO/evfn1X//1PPbYYznjjDOO6ZxFixZlz549vdvWrVsrPCUAUE0DK/nkZ5xxRgYMGJAdO3b02b9jx440NDS85/jvfe972bJlS6688srefd3d3T8YdODAbN68Oeedd16fc+rq6lJXV1eB6QGAk1FFV14GDRqUKVOmZNWqVb37uru7s2rVqkyfPv09x48bNy7f/e53s3Hjxt7tl37pl/JzP/dz2bhxoz8JAQCVXXlJkra2tsydOzfNzc2ZOnVqli1bln379mXevHlJkuuvvz5nnXVW2tvbM3jw4EycOLHP+cOHD0+S9+wHAE5NFY+Xa6+9Nm+++WbuuOOOdHZ2ZvLkyVmxYkXvTbxvvPFGamtPqltvAICTWE1PT09PtYf4MHV1dWXYsGHZs2dP6uvrqz0OP6amhd+s9gjvsWXp7GqPANDvHM/vb0seAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARREvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARREvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFEGVnsAAODImhZ+s9ojvMeWpbOren0rLwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARREvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFCUExIvDz30UJqamjJ48OBMmzYta9euPeKxjz32WC655JKcfvrpOf3009PS0nLU4wGAU0vF4+Xpp59OW1tbFi9enA0bNmTSpEmZOXNmdu7cedjjn3/++Xz2s5/NP/3TP6WjoyONjY257LLLsm3btkqPCgAUoOLxct9992XBggWZN29eJkyYkEceeSSnnXZannjiicMe/+STT+a3f/u3M3ny5IwbNy5/+qd/mu7u7qxatarSowIABahovLzzzjtZv359WlpafnTB2tq0tLSko6PjmJ5j//79OXjwYEaMGHHYxw8cOJCurq4+GwDQf1U0Xnbt2pVDhw5l9OjRffaPHj06nZ2dx/Qcv/u7v5sxY8b0CaD/v/b29gwbNqx3a2xs/LHnBgBOXif1u42WLl2ap556Ks8880wGDx582GMWLVqUPXv29G5bt249wVMCACfSwEo++RlnnJEBAwZkx44dffbv2LEjDQ0NRz33nnvuydKlS/OP//iP+dSnPnXE4+rq6lJXV/ehzAsAnPwquvIyaNCgTJkypc/Ntj+8+Xb69OlHPO+rX/1qlixZkhUrVqS5ubmSIwIAhanoykuStLW1Ze7cuWlubs7UqVOzbNmy7Nu3L/PmzUuSXH/99TnrrLPS3t6eJLn77rtzxx13ZPny5Wlqauq9N+ZjH/tYPvaxj1V6XADgJFfxeLn22mvz5ptv5o477khnZ2cmT56cFStW9N7E+8Ybb6S29kcLQA8//HDeeeed/Mqv/Eqf51m8eHHuvPPOSo8Lp7Smhd+s9gjvsWXp7GqPAJxkKh4vSdLa2prW1tbDPvb888/3+XnLli2VHwgAKNZJ/W4jAID/l3gBAIoiXgCAoogXAKAoJ+SG3f7EuzEAoLqsvAAARREvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARREvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARREvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARRlY7QEAKEvTwm9We4T32LJ0drVH4ASy8gIAFEW8AABFES8AQFHECwBQFPECABRFvAAARTkh8fLQQw+lqakpgwcPzrRp07J27dqjHv/Xf/3XGTduXAYPHpzzzz8/f//3f38ixgQAClDxeHn66afT1taWxYsXZ8OGDZk0aVJmzpyZnTt3Hvb4f/mXf8lnP/vZzJ8/P9/5zndy1VVX5aqrrsorr7xS6VEBgAJUPF7uu+++LFiwIPPmzcuECRPyyCOP5LTTTssTTzxx2OP/6I/+KLNmzcrtt9+e8ePHZ8mSJbnwwgvz4IMPVnpUAKAAFY2Xd955J+vXr09LS8uPLlhbm5aWlnR0dBz2nI6Ojj7HJ8nMmTOPePyBAwfS1dXVZwMA+q+Kfj3Arl27cujQoYwePbrP/tGjR2fTpk2HPaezs/Owx3d2dh72+Pb29nz5y1/+cAY+BqV+BHWpH+ft9f7weL1PrGN5LUud2/+TD09/fr0rqfh3Gy1atCh79uzp3bZu3VrtkQCACqroyssZZ5yRAQMGZMeOHX3279ixIw0NDYc9p6Gh4biOr6urS11d3YczMABw0qvoysugQYMyZcqUrFq1qndfd3d3Vq1alenTpx/2nOnTp/c5PklWrlx5xOMBgFNLRVdekqStrS1z585Nc3Nzpk6dmmXLlmXfvn2ZN29ekuT666/PWWedlfb29iTJzTffnEsvvTT33ntvZs+enaeeeiovv/xyHn300UqPCgAUoOLxcu211+bNN9/MHXfckc7OzkyePDkrVqzovSn3jTfeSG3tjxaALrrooixfvjy///u/ny9+8Yv5qZ/6qXzjG9/IxIkTKz0qAFCAisdLkrS2tqa1tfWwjz3//PPv2XfNNdfkmmuuqfBUAECJin+3EQBwahEvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARREvAEBRxAsAUBTxAgAURbwAAEURLwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARREvAEBRxAsAUBTxAgAURbwAAEUZWO0BAH5cW5bOrvYIwAlk5QUAKIp4AQCKIl4AgKKIFwCgKOIFACiKeAEAiiJeAICiiBcAoCjiBQAoingBAIoiXgCAoogXAKAo4gUAKIp4AQCKIl4AgKKIFwCgKOIFACiKeAEAilLReNm9e3fmzJmT+vr6DB8+PPPnz8/bb7991OM///nP5xOf+ESGDBmSsWPH5gtf+EL27NlTyTEBgIJUNF7mzJmTV199NStXrsxzzz2XNWvW5MYbbzzi8du3b8/27dtzzz335JVXXsmf//mfZ8WKFZk/f34lxwQACjKwUk/82muvZcWKFVm3bl2am5uTJA888ECuuOKK3HPPPRkzZsx7zpk4cWK+/vWv9/583nnn5a677srnPve5vPvuuxk4sGLjAgCFqNjKS0dHR4YPH94bLknS0tKS2travPTSS8f8PHv27El9fb1wAQCSVHDlpbOzM6NGjep7sYEDM2LEiHR2dh7Tc+zatStLliw56p+aDhw4kAMHDvT+3NXV9cEGBgCKcNwrLwsXLkxNTc1Rt02bNv3Yg3V1dWX27NmZMGFC7rzzziMe197enmHDhvVujY2NP/a1AYCT13GvvNx222254YYbjnrMueeem4aGhuzcubPP/nfffTe7d+9OQ0PDUc/fu3dvZs2alaFDh+aZZ57JRz7ykSMeu2jRorS1tfX+3NXV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8m//9m+59957c/rpp1d7tH7p7rvvzsMPP5wHH3wwr732Wu6+++589atfzQMPPFDt0U5J3ip9gk2bNi2f/vSn8+CDDyb5wXcxNTY25vOf/3wWLlxY5en6tzfffDOjRo3Kt771rXzmM5+p9jj91ttvv50LL7wwf/zHf5w//MM/zOTJk7Ns2bJqj9XvLFy4MC+++GL++Z//udqjnBJ+8Rd/MaNHj87jjz/eu+/qq6/OkCFD8pd/+ZdVnOzUZOXlBHrnnXeyfv36tLS09O6rra1NS0tLOjo6qjjZqWHPnj1JkhEjRlR5kv7tpptuyuzZs/v8P+fD9+yzz6a5uTnXXHNNRo0alQsuuCCPPfZYtcfqty666KKsWrUqr7/+epLkX//1X/PCCy/k8ssvr/Jkp6Z+98WMJ7Ndu3bl0KFDGT16dJ/9o0ePzqZNm6o01amhu7s7t9xySy6++OJMnDix2uP0W0899VQ2bNiQdevWVXuUfu/73/9+Hn744bS1teWLX/xi1q1bly984QsZNGhQ5s6dW+3x+p2FCxemq6sr48aNy4ABA3Lo0KHcddddmTNnTrVHOyWJF04JN910U1555ZW88MIL1R6l39q6dWtuvvnmrFy5MoMHD672OP1ed3d3mpub85WvfCVJcsEFF+SVV17JI488Il4q4K/+6q/y5JNPZvny5fnkJz+ZjRs35pZbbsmYMWO83lUgXk6gM844IwMGDMiOHTv67N+xY0caGhqqNFX/19ramueeey5r1qzJ2WefXe1x+q3169dn586dufDCC3v3HTp0KGvWrMmDDz6YAwcOZMCAAVWcsH8588wzM2HChD77xo8fn69//etVmqh/u/3227Nw4cJcd911SZLzzz8///Vf/5X29nbxUgXueTmBBg0alClTpmTVqlW9+7q7u7Nq1apMnz69ipP1Tz09PWltbc0zzzyT1atX55xzzqn2SP3ajBkz8t3vfjcbN27s3ZqbmzNnzpxs3LhRuHzILr744ve89f/111/PT/zET1Rpov5t//79qa3t+ytzwIAB6e7urtJEpzYrLydYW1tb5s6dm+bm5kydOjXLli3Lvn37Mm/evGqP1u/cdNNNWb58ef72b/82Q4cOTWdnZ5Jk2LBhGTJkSJWn63+GDh36nvuJPvrRj+bjH/+4+4wq4NZbb81FF12Ur3zlK/nVX/3VrF27No8++mgeffTRao/WL1155ZW56667Mnbs2Hzyk5/Md77zndx33335jd/4jWqPdmrq4YR74IEHesaOHdszaNCgnqlTp/Z8+9vfrvZI/VKSw25/9md/Vu3RThmXXnppz80331ztMfqtv/u7v+uZOHFiT11dXc+4ceN6Hn300WqP1G91dXX13HzzzT1jx47tGTx4cM+5557b83u/93s9Bw4cqPZopySf8wIAFMU9LwBAUcQLAFAU8QIAFEW8AABFES8AQFHECwBQFPECABRFvAAARREvAEBRxAsAUBTxAgAURbwAAEX5/wD4YExF24ZDNwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#logistic regression feature importance\n",
    "from sklearn.datasets import make_classification\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from matplotlib import pyplot\n",
    "#define dataset\n",
    "X, y= make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)\n",
    "#Define the model\n",
    "model =LogisticRegression()\n",
    "#fit the model\n",
    "model.fit(X,y)\n",
    "#get importance\n",
    "importance = model.coef_[0]\n",
    "#summarize the feature importance\n",
    "for i, v in enumerate(importance):\n",
    "    print ('Feature: %0d , Score: %.5f' %(i, v))\n",
    "#plot feature importance\n",
    "pyplot.bar([x for x in range (len(importance))], importance)\n",
    "pyplot.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
