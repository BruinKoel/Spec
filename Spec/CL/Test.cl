struct Node4 {
  float w1;
  float w2;
  float w3;
  float w4;

  float bias;

  long lastActivation;
  long activityCounter;
  double totalActivation;
  float decayFactor;

  float state;
};

__kernel void Cycle(__global int *indices, __global struct Node4 *nodes,
                    const long wave, const int size) {
  int gid = get_global_id(0);
  int globalSize = get_global_size(0);
  int localSize = get_local_size(0);
  int end = gid + localSize;

  for (int i = gid; i < end; i++) {
    struct Node4 node = nodes[indices[i]];
    int decay = wave - node.lastActivation;

    if (decay == 0) {
      continue;
    }
    int up = (gid - size) % globalSize;
    int down = (gid + size) % globalSize;
    int right = (gid + 1) % globalSize;
    int left = (gid - 1) % globalSize;

    node.state = (node.state * Math.pow(decayFactor, decay)) +
                node.bias +
                 nodes[up].state * node.w1 + 
                 nodes[right].state * node.w2 +
                 nodes[down].state * node.w3 + 
                 nodes[left].state * node.w4;

    node.lastActivation = wave;
    node.activityCounter++;
    node.totalActivation += node.state;

    nodes[indices[i]] = node;
  }
}

__kernel void add_arrays(__global float *a, __global float *b,
                         __global float *result, const unsigned int count) {
  int gid = get_global_id(0);

  if (gid < count) {
    result[gid] = a[gid] + b[gid];
  }
}
