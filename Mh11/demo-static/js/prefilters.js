// FIR (h0,h1,h2) and IIR2 (g0,g1,g2 -- the factored all-pole form, NOT the
// undecomposed h-coefficients) dQI prefilter coefficients, transcribed
// verbatim from each family's prefilter_dict in the corresponding module:
//   OneDrive-Personal/work/python/{Mh11,Mh20,Vhex}.py

export const prefilters = {
    Mh11: {
        FIR: [37 / 12, -7 / 16, 13 / 144],
        IIR2: [0.768044002012887, 0.23647686165263507, -0.004520863665522072],
    },
    Mh20: {
        FIR: [37 / 20, -41 / 240, 7 / 240],
        IIR2: [0.8919696445605764, 0.11270166537925831, -0.004671309939834724],
    },
    Vhex4: {
        FIR: [4771 / 2430, -1891 / 9720, 991 / 29160],
        IIR2: [0.8800327721147185, 0.12466293155086716, -0.004695703665585662],
    },
};
