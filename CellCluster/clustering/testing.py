
# This function creates synthetic images: clouds of points that are confined within a squared of defined size
def box(d,b,c):
  box1 = np.random.rand(2000,2) - 0.5
  box1 = d*box1
  box1[:,0] = box1[:,0] + b
  box1[:,1] = box1[:,1] + c
  return box1
