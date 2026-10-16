// FIR (h0,h1,h2) dQI prefilter coefficients, transcribed verbatim from each
// family's prefilter_dict in OneDrive-Personal/work/python/{Mh11,Mh20,Vhex}.py
// (same source as ../../web/js/prefilters.js). IIR is intentionally omitted
// here: this demo only ever runs on the GPU, and the factored IIR prefilter
// requires a CPU-side sequential recursion (see ../../web/js/iir2.js) --
// which is exactly why this animation demo offers no IIR option at all.

export const prefilters = {
    Mh11: { FIR: [37 / 12, -7 / 16, 13 / 144] },
    Mh20: { FIR: [37 / 20, -41 / 240, 7 / 240] },
    Vhex4: { FIR: [4771 / 2430, -1891 / 9720, 991 / 29160] },
};
