using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace Spec.CL.Structs
{
    [StructLayout(LayoutKind.Sequential,Size =52,Pack = 4)]
    public struct Node4
    {
        public long lastActivation;
        public long activityCounter;
        public double totalActivation;

        public float w1;
        public float w2;
        public float w3;
        public float w4;

        public float bias;
        public float state;
        public float decayFactor;
    }
}
