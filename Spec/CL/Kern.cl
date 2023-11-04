struct Node4 {
        long lastActivation;
        long activityCounter;
        double totalActivation;

        float w1;
        float w2;
        float w3;
        float w4;

        float bias;
        float state;
        float decayFactor;
};

__kernel void Cycle( __global struct Node4 *nodes,__global int *indices,
                    const long wave, const int size) {
  int gid = get_global_id(0);
  int globalSize = get_global_size(0);
  int localSize = get_local_size(0);
  int end = gid + localSize;

  for (int i = gid; i < end; i += 1) {
    struct Node4 node = nodes[indices[i]];
    int decay = wave - node.lastActivation;

    if (decay == 0) {
      continue;
    }
    int up = (gid - size) % globalSize;
    int down = (gid + size) % globalSize;
    int right = (gid + 1) % globalSize;
    int left = (gid - 1) % globalSize;

    node.state = (node.state * pow(node.decayFactor, decay + 1)) + node.bias +
                 nodes[up].state * node.w1 + nodes[right].state * node.w2 +
                 nodes[down].state * node.w3 + nodes[left].state * node.w4;

    node.lastActivation = wave;
    node.activityCounter += 1;
    node.totalActivation += node.state;

    nodes[indices[i]] = node;
  }
}
