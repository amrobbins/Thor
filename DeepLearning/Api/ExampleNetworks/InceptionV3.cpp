

//[type]          [patch size/stride] [output size]  [depth]  [#1×1]  [#3×3 reduce]   [#3×3]  [#5×5 reduce]   [#5×5]  [pool proj] [params]

// convolution     7×7/2               112×112×64      1                                                                           2.7K
// max pool        3×3/2               56×56×64        0
// convolution     3×3/1               56×56×192       2               64              192     112K                                360M
// max pool        3×3/2               28×28×192       0
// inception (3a)                      28×28×256       2       64      96              128     16              32      32          159K
// inception (3b)                      28×28×480       2       128     128             192     32              96      64          380K
// max pool 3×3/2                      14×14×480       0
// inception (4a)                      14×14×512       2       192     96              208     16              48      64          364K
// inception (4b)                      14×14×512       2       160     112             224     24              64      64          437K
// inception (4c)                      14×14×512       2       128     128             256     24              64      64          463K
// inception (4d)                      14×14×528       2       112     144             288     32              64      64          580K
// inception (4e)                      14×14×832       2       256     160             320     32              128     128         840K
// max pool 3×3/2                      7×7×832         0
// inception (5a)                      7×7×832         2       256     160             320     32              128     128         1072K
// inception (5b)                      7×7×1024        2       384     192             384     48              128     128         1388K
// avg pool 7×7/1                      1×1×1024        0
// dropout (40%)                       1×1×1024        0
// linear                              1×1×1000        1                                                                           1000K
// softmax                             1×1×1000        0
