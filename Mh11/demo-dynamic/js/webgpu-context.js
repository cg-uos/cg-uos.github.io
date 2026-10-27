// Device init, plus a *persistent*-resource compute-kernel wrapper.
//
// Recreating the shader module, pipeline, and every buffer on every call
// would be fine for a one-shot action, but far too much per-frame overhead
// for a ~30-60fps animation loop. `GPUKernel` instead builds those once
// (`configure`) and, every frame, only writes new input data and reads
// back the result (`run`), via buffers it keeps alive until `destroy()`.

export async function initWebGPU() {
    if (!('gpu' in navigator)) {
        throw new Error(
            'WebGPU is not supported in this browser. Try a recent desktop Chrome or Edge.'
        );
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('No suitable WebGPU adapter was found on this system.');
    }
    return adapter.requestDevice();
}

export class GPUKernel {
    constructor(device, wgslSource) {
        this.device = device;
        this.module = device.createShaderModule({ code: wgslSource });
        this.inputBuffers = [];
        this.outputBuffers = [];
        this.stagingBuffers = [];
        this.pipeline = null;
        this.bindGroup = null;
    }

    // inputByteLengths/outputByteLengths: byte length of each binding, in
    // binding-index order (inputs first at 0..n-1, outputs right after).
    configure(inputByteLengths, outputByteLengths) {
        this.inputBuffers = inputByteLengths.map((size) =>
            this.device.createBuffer({
                size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            })
        );
        this.outputBuffers = outputByteLengths.map((size) =>
            this.device.createBuffer({
                size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            })
        );
        this.stagingBuffers = outputByteLengths.map((size) =>
            this.device.createBuffer({
                size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            })
        );
        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.module, entryPoint: 'main' },
        });
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                ...this.inputBuffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
                ...this.outputBuffers.map((buffer, i) => ({
                    binding: this.inputBuffers.length + i,
                    resource: { buffer },
                })),
            ],
        });
    }

    // inputs: array of typed arrays, one per configured input buffer.
    // dispatch: [x,y,z] workgroup counts. Returns one Float32Array per
    // configured output buffer.
    async run(inputs, dispatch) {
        inputs.forEach((data, i) => this.device.queue.writeBuffer(this.inputBuffers[i], 0, data));

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(...dispatch);
        pass.end();
        this.outputBuffers.forEach((buf, i) => {
            encoder.copyBufferToBuffer(buf, 0, this.stagingBuffers[i], 0, buf.size);
        });
        this.device.queue.submit([encoder.finish()]);

        const results = [];
        for (const staging of this.stagingBuffers) {
            await staging.mapAsync(GPUMapMode.READ);
            results.push(new Float32Array(staging.getMappedRange().slice(0)));
            staging.unmap();
        }
        return results;
    }

    destroy() {
        for (const buf of [...this.inputBuffers, ...this.outputBuffers, ...this.stagingBuffers]) {
            buf.destroy();
        }
        this.inputBuffers = [];
        this.outputBuffers = [];
        this.stagingBuffers = [];
    }
}
