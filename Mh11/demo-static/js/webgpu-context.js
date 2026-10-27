// Minimal WebGPU device init + a generic compute-dispatch/readback helper

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

// Runs a compute shader with `inputs` (an array of Float32Array/Int32Array,
// bound read-only at bindings 0..inputs.length-1) and `outputByteLengths` (an
// array of byte lengths, bound read_write immediately after the inputs).
// `dispatch` is [x,y,z] workgroup counts. Returns one Float32Array per output.
export async function runCompute(device, wgslSource, inputs, outputByteLengths, dispatch) {
    const module = device.createShaderModule({ code: wgslSource });

    const inputBuffers = inputs.map((data) => {
        const buf = device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buf, 0, data);
        return buf;
    });

    const outputBuffers = outputByteLengths.map((byteLength) =>
        device.createBuffer({
            size: byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        })
    );

    const bindGroupEntries = [
        ...inputBuffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
        ...outputBuffers.map((buffer, i) => ({
            binding: inputBuffers.length + i,
            resource: { buffer },
        })),
    ];

    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindGroupEntries,
    });

    const stagingBuffers = outputBuffers.map((buf) =>
        device.createBuffer({
            size: buf.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        })
    );

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatch);
    pass.end();
    outputBuffers.forEach((buf, i) => {
        encoder.copyBufferToBuffer(buf, 0, stagingBuffers[i], 0, buf.size);
    });
    device.queue.submit([encoder.finish()]);

    const results = [];
    for (const staging of stagingBuffers) {
        await staging.mapAsync(GPUMapMode.READ);
        results.push(new Float32Array(staging.getMappedRange().slice(0)));
        staging.unmap();
    }

    for (const buf of [...inputBuffers, ...outputBuffers, ...stagingBuffers]) {
        buf.destroy();
    }

    return results;
}
