import numpy as np
import copy
def softmax_custom( logits ):
    logits = logits - np.max( logits, axis= 1, keepdims=True )
    exp_scores = np.exp(logits) 
    prob = exp_scores / np.sum(  exp_scores , axis = 1, keepdims=True)
    return prob

def cross_entropy_loss_custom(prob, y_label):
    B, D = prob.shape
    correct_probs = -np.log( np.take_along_axis(prob, y_label[..., None] , axis = -1)).squeeze(-1)
    loss = np.sum(correct_probs) / B
    return loss

def forward_backward_custom(logits, y_label):
    B, D = logits.shape
    prob = softmax(logits)
    loss = cross_entropy_loss_custom(prob, y_label)

    dlogits = copy.deepcopy(prob)
    dlogits[np.arange(B), y_label] -= 1
    
    return loss, dlogits

logits = np.array(
    [
        [2., 1., 0.1, .5],
        [1., 3., 0.5, 2.],
        [0.1, 0.2, 1.5, 1.]
    ]
)
y_true = np.array( [0, 1, 2] )
loss, grad = forward_backward_custom(logits, y_true)
print("Loss: ", loss)
print("Grad: ", grad)