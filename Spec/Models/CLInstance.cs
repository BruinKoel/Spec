using OpenCL.Net;
using Spec.CL.Structs;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using CLProgram = OpenCL.Net.Program;
using DeviceType = OpenCL.Net.DeviceType;
using Platform = OpenCL.Net.Platform;


namespace Spec.Models
{
    public class CLInstance
    {
        public static string AssemblyDirectory
        {
            get
            {
                string codeBase = Assembly.GetExecutingAssembly().Location;
                UriBuilder uri = new UriBuilder(codeBase);
                string path = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(path);
            }
        }

        private ErrorCode CurrentError;
        private Kernel kernel;
        private Context context;
        CommandQueue commandQueue;
        private IMem NetWorkRef;
        private Node4[] Network;
        private IMem IndexRef;
        int[] Indices;
        private int NetSize = 1024;
        private nint[] localWorkSize = { 256 };
        private nint[] globalWorkSize ;


        public CLInstance()
        {
            Platform[] platforms = Cl.GetPlatformIDs(out CurrentError);
            CheckError(CurrentError);
            DeviceType deviceType = DeviceType.Gpu;

            context = Cl.CreateContextFromType(null, deviceType, null, 0, out CurrentError);
            CheckError(CurrentError);
            commandQueue = Cl.CreateCommandQueue(context, Cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, out CurrentError)[0], CommandQueueProperties.None, out CurrentError);
            CheckError(CurrentError);

            // Load the OpenCL kernel from a .cl file
            // Load the OpenCL kernel source code from your .cl file
            string kernelSource = System.IO.File.ReadAllText(ProgramLocation);

            // Create an OpenCL program
            CLProgram program = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out CurrentError);
            CheckError(CurrentError);

            // Build the program
            Cl.BuildProgram(program, 1, new[] { Cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, out CurrentError)[0] }, null, null, IntPtr.Zero);
            CheckError(CurrentError);

            // Create a kernel
            kernel = Cl.CreateKernel(program, "Cycle", out CurrentError);
            CheckError(CurrentError);// Change to your kernel function name
        }
        public void CheckError(ErrorCode error)
        {
            if (error != ErrorCode.Success)
            {
                Debug.WriteLine(error.ToString());
            }
        }

        public void Launch(Node4[] net)
        {
            Indices = GenerateIndices(net);
            IndexRef = Cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.AllocHostPtr, sizeof(int) * Indices.Length, out CurrentError);
            CheckError(CurrentError);
            Network = net;
            NetWorkRef = Cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.AllocHostPtr, Marshal.SizeOf<Node4>() * net.Length, out CurrentError);

            globalWorkSize = new nint[] { Indices.Length };


            Cl.SetKernelArg(kernel, 0, NetWorkRef);
            Cl.SetKernelArg(kernel, 1, IndexRef);

            Cl.SetKernelArg(kernel, 3, NetSize);

            Cl.EnqueueWriteBuffer(commandQueue, NetWorkRef, Bool.True, IntPtr.Zero, Marshal.SizeOf<Node4>() * net.Length, Network, 0, null, out Event writeEvent);
            Cl.EnqueueWriteBuffer(commandQueue, IndexRef, Bool.True, IntPtr.Zero, sizeof(int) * Indices.Length, Indices, 0, null, out Event writeEvent2);
           

            Cycle();

            Cl.ReleaseMemObject(IndexRef);
            Cl.ReleaseMemObject(NetWorkRef);

        }

        public void Cycle(long count = 8)
        {
            for (long wave = 0; wave < count; wave++)
            {
                Cl.SetKernelArg(kernel, 2, wave);
                Cl.EnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, 0, null, out Event eventID);
                Cl.WaitForEvents(1,new[] { eventID });

            }
        }

        

        public int[] GenerateIndices(Node4[] nodes, int depth = 4)
        {
            int[] result = new int[depth * nodes.Length];
            int[][] temp = new int[depth][];

            Parallel.For(0, depth, (i) =>
            {
                temp[i] = new int[nodes.Length];
                Random random = new Random();
                for (int o = 0; o < nodes.Length; o++)
                {
                    temp[i][o] = o;
                }
                MauiProgram.Shuffle(random, temp[i]);
                Array.ConstrainedCopy(temp[i], 0, result, i * nodes.Length, nodes.Length);
            });

            return result;
        }

        public Node4[] GenerateNet(int Size)
        {
            NetSize = Size;
            Node4[] nodes = new Node4[Size * Size];

            Parallel.For(0, nodes.Length, (i) =>
            {
                Random random = new Random();
                nodes[i] = new Node4()
                {
                    w1 = random.NextSingle(),
                    w2 = random.NextSingle(),
                    w3 = random.NextSingle(),
                    w4 = random.NextSingle(),
                    bias = random.NextSingle(),
                    decayFactor = random.NextSingle(),
                };
            });

            return nodes;

        }

        virtual public string ProgramLocation => Path.Combine(AssemblyDirectory, "CL\\Kern.cl");



    }


}
