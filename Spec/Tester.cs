using OpenCL.Net;

using Platform = OpenCL.Net.Platform;
using CLProgram = OpenCL.Net.Program;

using DeviceType = OpenCL.Net.DeviceType;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net.Http.Headers;
using System.Reflection;
using System.Diagnostics;

namespace Spec
{
    public class Tester
    {
        ErrorCode CurrentError;
        string KernelSources;
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
        public void Main()
        {
            KernelSources = Path.Combine(AssemblyDirectory, "CL");

            // Initialize OpenCL
            Platform[] platforms = Cl.GetPlatformIDs(out CurrentError);
            DeviceType deviceType = DeviceType.Gpu;

            Context context = Cl.CreateContextFromType(null, deviceType, null, 0, out CurrentError);
            CommandQueue commandQueue = Cl.CreateCommandQueue(context, Cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, out CurrentError)[0], CommandQueueProperties.None, out CurrentError);

            // Load the OpenCL kernel from a .cl file
            // Load the OpenCL kernel source code from your .cl file
            string kernelSource = System.IO.File.ReadAllText(Path.Combine(KernelSources, "Test.cl"));

            // Create an OpenCL program
            CLProgram program = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out CurrentError);

            // Build the program
            Cl.BuildProgram(program, 1, new[] { Cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, out CurrentError)[0] }, null, null, IntPtr.Zero);

            // Create a kernel
            Kernel kernel = Cl.CreateKernel(program, "add_arrays", out CurrentError); // Change to your kernel function name

            // Set kernel arguments
            int arraySize = 1024*1024; // Define the size of your arrays
            IMem aBuffer = Cl.CreateBuffer(context, MemFlags.ReadWrite, arraySize * sizeof(float), out CurrentError);
            IMem bBuffer = Cl.CreateBuffer(context, MemFlags.ReadWrite, arraySize * sizeof(float), out CurrentError);
            IMem resultBuffer = Cl.CreateBuffer(context, MemFlags.ReadWrite, arraySize * sizeof(float), out CurrentError);

            // Set the kernel arguments
            Cl.SetKernelArg(kernel, 0, aBuffer);
            Cl.SetKernelArg(kernel, 1, bBuffer);
            Cl.SetKernelArg(kernel, 2, resultBuffer);
            Cl.SetKernelArg(kernel, 3, arraySize);

            // Define the global and local work sizes
            nint[] globalWorkSize = new nint[] { arraySize };
            nint[] localWorkSize = null; // Set to null for automatic local work size

            // Generate random input data and send it to the device
            float[] aData = new float[arraySize];
            float[] bData = new float[arraySize];

            // Populate aData and bData with random values (you can use your own method for this)
            Random random = new Random();
            for (int i = 0; i < arraySize; i++)
            {
                aData[i] = (float)random.NextDouble();
                bData[i] = (float)random.NextDouble();
            }

            Cl.EnqueueWriteBuffer(commandQueue, aBuffer, Bool.True, IntPtr.Zero, arraySize * sizeof(float), aData, 0, null, out Event writeEvent);
            Cl.EnqueueWriteBuffer(commandQueue, bBuffer, Bool.True, IntPtr.Zero, arraySize * sizeof(float), bData, 0, null, out Event writeEvent2);

            // Enqueue the kernel for execution
            Cl.EnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, localWorkSize, 0, null, out Event eventID);

            // Wait for the kernel to finish
            Cl.WaitForEvents(1, new[] { eventID , writeEvent , writeEvent2 });

            // Read the result from the device
            float[] resultData = new float[arraySize];
            Cl.EnqueueReadBuffer(commandQueue, resultBuffer, Bool.True, IntPtr.Zero, arraySize * sizeof(float), resultData, 0, null, out Event readEvent);

            // Check the result
            for (int i = 0; i < arraySize; i++)
            {
                Debug.WriteLine("Result[{0}] = {1} = {3} = {2}", i, resultData[i], $"{aData[i]} + {bData[i]}",aData[i] + bData[i] );
            }

            // Cleanup and release resources
            Cl.ReleaseMemObject(aBuffer);
            Cl.ReleaseMemObject(bBuffer);
            Cl.ReleaseMemObject(resultBuffer);
            Cl.ReleaseKernel(kernel);
            Cl.ReleaseProgram(program);
            Cl.ReleaseCommandQueue(commandQueue);
            Cl.ReleaseContext(context);

        }
    }
    }
