ý
Ö!Ż!
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
ë
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
Equal
x"T
y"T
z
"
Ttype:
2	

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
Ą
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype

ImageSummary
tag
tensor"T
summary"

max_imagesint(0"
Ttype0:
2"'
	bad_colortensorB:˙  ˙
b
InitializeTableV2
table_handle
keys"Tkey
values"Tval"
Tkeytype"
Tvaltype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ô
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.6.02v1.6.0-0-gd2e24b6039Ń

-global_step/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@global_step*
dtype0*
_output_shapes
: 

#global_step/Initializer/zeros/ConstConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Ě
global_step/Initializer/zerosFill-global_step/Initializer/zeros/shape_as_tensor#global_step/Initializer/zeros/Const*
T0	*

index_type0*
_class
loc:@global_step*
_output_shapes
: 

global_step
VariableV2*
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name 
˛
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
~
PlaceholderPlaceholder*$
shape:˙˙˙˙˙˙˙˙˙  *
dtype0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Q

images/tagConst*
valueB Bimages*
dtype0*
_output_shapes
: 

imagesImageSummary
images/tagPlaceholder*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: *

max_images
J
ConstConst*
valueB
 *  ČB*
dtype0*
_output_shapes
: 
L
Const_1Const*
valueB
 *
×Ł;*
dtype0*
_output_shapes
: 
L
Const_2Const*
valueB
 *  zD*
dtype0*
_output_shapes
: 
L
Const_3Const*
valueB
 *{n?*
dtype0*
_output_shapes
: 
_
ExponentialDecay/CastCastglobal_step/read*

SrcT0	*
_output_shapes
: *

DstT0
d
ExponentialDecay/truedivRealDivExponentialDecay/CastConst_2*
T0*
_output_shapes
: 
_
ExponentialDecay/PowPowConst_3ExponentialDecay/truediv*
T0*
_output_shapes
: 
W
ExponentialDecayMulConst_1ExponentialDecay/Pow*
T0*
_output_shapes
: 
`
learning_rate/tagsConst*
valueB Blearning_rate*
dtype0*
_output_shapes
: 
e
learning_rateScalarSummarylearning_rate/tagsExponentialDecay*
T0*
_output_shapes
: 
Ă
;hidden_layer_0/conv/kernel/Initializer/random_uniform/shapeConst*%
valueB"             *-
_class#
!loc:@hidden_layer_0/conv/kernel*
dtype0*
_output_shapes
:
­
9hidden_layer_0/conv/kernel/Initializer/random_uniform/minConst*
valueB
 *ž*-
_class#
!loc:@hidden_layer_0/conv/kernel*
dtype0*
_output_shapes
: 
­
9hidden_layer_0/conv/kernel/Initializer/random_uniform/maxConst*
valueB
 *>*-
_class#
!loc:@hidden_layer_0/conv/kernel*
dtype0*
_output_shapes
: 

Chidden_layer_0/conv/kernel/Initializer/random_uniform/RandomUniformRandomUniform;hidden_layer_0/conv/kernel/Initializer/random_uniform/shape*
T0*-
_class#
!loc:@hidden_layer_0/conv/kernel*
seed2 *
dtype0*&
_output_shapes
: *

seed 

9hidden_layer_0/conv/kernel/Initializer/random_uniform/subSub9hidden_layer_0/conv/kernel/Initializer/random_uniform/max9hidden_layer_0/conv/kernel/Initializer/random_uniform/min*
T0*-
_class#
!loc:@hidden_layer_0/conv/kernel*
_output_shapes
: 
 
9hidden_layer_0/conv/kernel/Initializer/random_uniform/mulMulChidden_layer_0/conv/kernel/Initializer/random_uniform/RandomUniform9hidden_layer_0/conv/kernel/Initializer/random_uniform/sub*
T0*-
_class#
!loc:@hidden_layer_0/conv/kernel*&
_output_shapes
: 

5hidden_layer_0/conv/kernel/Initializer/random_uniformAdd9hidden_layer_0/conv/kernel/Initializer/random_uniform/mul9hidden_layer_0/conv/kernel/Initializer/random_uniform/min*
T0*-
_class#
!loc:@hidden_layer_0/conv/kernel*&
_output_shapes
: 
Í
hidden_layer_0/conv/kernel
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *-
_class#
!loc:@hidden_layer_0/conv/kernel*
	container *
shape: 

!hidden_layer_0/conv/kernel/AssignAssignhidden_layer_0/conv/kernel5hidden_layer_0/conv/kernel/Initializer/random_uniform*
use_locking(*
T0*-
_class#
!loc:@hidden_layer_0/conv/kernel*
validate_shape(*&
_output_shapes
: 
§
hidden_layer_0/conv/kernel/readIdentityhidden_layer_0/conv/kernel*
T0*-
_class#
!loc:@hidden_layer_0/conv/kernel*&
_output_shapes
: 
ą
:hidden_layer_0/conv/bias/Initializer/zeros/shape_as_tensorConst*
valueB: *+
_class!
loc:@hidden_layer_0/conv/bias*
dtype0*
_output_shapes
:
˘
0hidden_layer_0/conv/bias/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@hidden_layer_0/conv/bias*
dtype0*
_output_shapes
: 

*hidden_layer_0/conv/bias/Initializer/zerosFill:hidden_layer_0/conv/bias/Initializer/zeros/shape_as_tensor0hidden_layer_0/conv/bias/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@hidden_layer_0/conv/bias*
_output_shapes
: 
ą
hidden_layer_0/conv/bias
VariableV2*
dtype0*
_output_shapes
: *
shared_name *+
_class!
loc:@hidden_layer_0/conv/bias*
	container *
shape: 
ę
hidden_layer_0/conv/bias/AssignAssignhidden_layer_0/conv/bias*hidden_layer_0/conv/bias/Initializer/zeros*
T0*+
_class!
loc:@hidden_layer_0/conv/bias*
validate_shape(*
_output_shapes
: *
use_locking(

hidden_layer_0/conv/bias/readIdentityhidden_layer_0/conv/bias*
T0*+
_class!
loc:@hidden_layer_0/conv/bias*
_output_shapes
: 
r
!hidden_layer_0/conv/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ú
hidden_layer_0/conv/Conv2DConv2DPlaceholderhidden_layer_0/conv/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙   *
	dilations

˛
hidden_layer_0/conv/BiasAddBiasAddhidden_layer_0/conv/Conv2Dhidden_layer_0/conv/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙   
h
#hidden_layer_0/conv/LeakyRelu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
¤
!hidden_layer_0/conv/LeakyRelu/mulMul#hidden_layer_0/conv/LeakyRelu/alphahidden_layer_0/conv/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙   
Ş
%hidden_layer_0/conv/LeakyRelu/MaximumMaximum!hidden_layer_0/conv/LeakyRelu/mulhidden_layer_0/conv/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙   
m
hidden_layer_0/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:

hidden_layer_0/MeanMean%hidden_layer_0/conv/LeakyRelu/Maximumhidden_layer_0/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
r
hidden_layer_0/weights/tagsConst*'
valueB Bhidden_layer_0/weights*
dtype0*
_output_shapes
: 
z
hidden_layer_0/weightsScalarSummaryhidden_layer_0/weights/tagshidden_layer_0/Mean*
T0*
_output_shapes
: 
f
!hidden_layer_0/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ż
"hidden_layer_0/zero_fraction/EqualEqual%hidden_layer_0/conv/LeakyRelu/Maximum!hidden_layer_0/zero_fraction/zero*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙   

!hidden_layer_0/zero_fraction/CastCast"hidden_layer_0/zero_fraction/Equal*

SrcT0
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙   *

DstT0
{
"hidden_layer_0/zero_fraction/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             
Ž
!hidden_layer_0/zero_fraction/MeanMean!hidden_layer_0/zero_fraction/Cast"hidden_layer_0/zero_fraction/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
t
hidden_layer_0/sparsity/tagsConst*
_output_shapes
: *(
valueB Bhidden_layer_0/sparsity*
dtype0

hidden_layer_0/sparsityScalarSummaryhidden_layer_0/sparsity/tags!hidden_layer_0/zero_fraction/Mean*
T0*
_output_shapes
: 
o
hidden_layer_0/conv_1/tagConst*
dtype0*
_output_shapes
: *&
valueB Bhidden_layer_0/conv_1

hidden_layer_0/conv_1HistogramSummaryhidden_layer_0/conv_1/tag%hidden_layer_0/conv/LeakyRelu/Maximum*
T0*
_output_shapes
: 
Ú
hidden_layer_1/pool/MaxPoolMaxPool%hidden_layer_0/conv/LeakyRelu/Maximum*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID
m
hidden_layer_1/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:

hidden_layer_1/MeanMeanhidden_layer_1/pool/MaxPoolhidden_layer_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
r
hidden_layer_1/weights/tagsConst*'
valueB Bhidden_layer_1/weights*
dtype0*
_output_shapes
: 
z
hidden_layer_1/weightsScalarSummaryhidden_layer_1/weights/tagshidden_layer_1/Mean*
T0*
_output_shapes
: 
f
!hidden_layer_1/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ľ
"hidden_layer_1/zero_fraction/EqualEqualhidden_layer_1/pool/MaxPool!hidden_layer_1/zero_fraction/zero*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

!hidden_layer_1/zero_fraction/CastCast"hidden_layer_1/zero_fraction/Equal*

SrcT0
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *

DstT0
{
"hidden_layer_1/zero_fraction/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0
Ž
!hidden_layer_1/zero_fraction/MeanMean!hidden_layer_1/zero_fraction/Cast"hidden_layer_1/zero_fraction/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
t
hidden_layer_1/sparsity/tagsConst*(
valueB Bhidden_layer_1/sparsity*
dtype0*
_output_shapes
: 

hidden_layer_1/sparsityScalarSummaryhidden_layer_1/sparsity/tags!hidden_layer_1/zero_fraction/Mean*
T0*
_output_shapes
: 
o
hidden_layer_1/pool_1/tagConst*&
valueB Bhidden_layer_1/pool_1*
dtype0*
_output_shapes
: 

hidden_layer_1/pool_1HistogramSummaryhidden_layer_1/pool_1/taghidden_layer_1/pool/MaxPool*
T0*
_output_shapes
: 
Ç
=hidden_layer_2/conv_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"              */
_class%
#!loc:@hidden_layer_2/conv_1/kernel*
dtype0*
_output_shapes
:
ą
;hidden_layer_2/conv_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ěŃ˝*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel*
dtype0
ą
;hidden_layer_2/conv_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *ěŃ=*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel*
dtype0

Ehidden_layer_2/conv_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform=hidden_layer_2/conv_1/kernel/Initializer/random_uniform/shape*
T0*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel*
seed2 *
dtype0*&
_output_shapes
:  *

seed 

;hidden_layer_2/conv_1/kernel/Initializer/random_uniform/subSub;hidden_layer_2/conv_1/kernel/Initializer/random_uniform/max;hidden_layer_2/conv_1/kernel/Initializer/random_uniform/min*
T0*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel*
_output_shapes
: 
¨
;hidden_layer_2/conv_1/kernel/Initializer/random_uniform/mulMulEhidden_layer_2/conv_1/kernel/Initializer/random_uniform/RandomUniform;hidden_layer_2/conv_1/kernel/Initializer/random_uniform/sub*&
_output_shapes
:  *
T0*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel

7hidden_layer_2/conv_1/kernel/Initializer/random_uniformAdd;hidden_layer_2/conv_1/kernel/Initializer/random_uniform/mul;hidden_layer_2/conv_1/kernel/Initializer/random_uniform/min*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel*&
_output_shapes
:  *
T0
Ń
hidden_layer_2/conv_1/kernel
VariableV2*
shared_name */
_class%
#!loc:@hidden_layer_2/conv_1/kernel*
	container *
shape:  *
dtype0*&
_output_shapes
:  

#hidden_layer_2/conv_1/kernel/AssignAssignhidden_layer_2/conv_1/kernel7hidden_layer_2/conv_1/kernel/Initializer/random_uniform*
use_locking(*
T0*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel*
validate_shape(*&
_output_shapes
:  
­
!hidden_layer_2/conv_1/kernel/readIdentityhidden_layer_2/conv_1/kernel*
T0*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel*&
_output_shapes
:  
ľ
<hidden_layer_2/conv_1/bias/Initializer/zeros/shape_as_tensorConst*
valueB: *-
_class#
!loc:@hidden_layer_2/conv_1/bias*
dtype0*
_output_shapes
:
Ś
2hidden_layer_2/conv_1/bias/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@hidden_layer_2/conv_1/bias*
dtype0

,hidden_layer_2/conv_1/bias/Initializer/zerosFill<hidden_layer_2/conv_1/bias/Initializer/zeros/shape_as_tensor2hidden_layer_2/conv_1/bias/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@hidden_layer_2/conv_1/bias*
_output_shapes
: 
ľ
hidden_layer_2/conv_1/bias
VariableV2*
shared_name *-
_class#
!loc:@hidden_layer_2/conv_1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ň
!hidden_layer_2/conv_1/bias/AssignAssignhidden_layer_2/conv_1/bias,hidden_layer_2/conv_1/bias/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@hidden_layer_2/conv_1/bias*
validate_shape(*
_output_shapes
: 

hidden_layer_2/conv_1/bias/readIdentityhidden_layer_2/conv_1/bias*
T0*-
_class#
!loc:@hidden_layer_2/conv_1/bias*
_output_shapes
: 
t
#hidden_layer_2/conv_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

hidden_layer_2/conv_1/Conv2DConv2Dhidden_layer_1/pool/MaxPool!hidden_layer_2/conv_1/kernel/read*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
¸
hidden_layer_2/conv_1/BiasAddBiasAddhidden_layer_2/conv_1/Conv2Dhidden_layer_2/conv_1/bias/read*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
j
%hidden_layer_2/conv_1/LeakyRelu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ş
#hidden_layer_2/conv_1/LeakyRelu/mulMul%hidden_layer_2/conv_1/LeakyRelu/alphahidden_layer_2/conv_1/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
°
'hidden_layer_2/conv_1/LeakyRelu/MaximumMaximum#hidden_layer_2/conv_1/LeakyRelu/mulhidden_layer_2/conv_1/BiasAdd*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
m
hidden_layer_2/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:

hidden_layer_2/MeanMean'hidden_layer_2/conv_1/LeakyRelu/Maximumhidden_layer_2/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
r
hidden_layer_2/weights/tagsConst*
dtype0*
_output_shapes
: *'
valueB Bhidden_layer_2/weights
z
hidden_layer_2/weightsScalarSummaryhidden_layer_2/weights/tagshidden_layer_2/Mean*
_output_shapes
: *
T0
f
!hidden_layer_2/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ą
"hidden_layer_2/zero_fraction/EqualEqual'hidden_layer_2/conv_1/LeakyRelu/Maximum!hidden_layer_2/zero_fraction/zero*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

!hidden_layer_2/zero_fraction/CastCast"hidden_layer_2/zero_fraction/Equal*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *

DstT0*

SrcT0

{
"hidden_layer_2/zero_fraction/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Ž
!hidden_layer_2/zero_fraction/MeanMean!hidden_layer_2/zero_fraction/Cast"hidden_layer_2/zero_fraction/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
t
hidden_layer_2/sparsity/tagsConst*
_output_shapes
: *(
valueB Bhidden_layer_2/sparsity*
dtype0

hidden_layer_2/sparsityScalarSummaryhidden_layer_2/sparsity/tags!hidden_layer_2/zero_fraction/Mean*
_output_shapes
: *
T0
s
hidden_layer_2/conv_1_1/tagConst*
_output_shapes
: *(
valueB Bhidden_layer_2/conv_1_1*
dtype0

hidden_layer_2/conv_1_1HistogramSummaryhidden_layer_2/conv_1_1/tag'hidden_layer_2/conv_1/LeakyRelu/Maximum*
T0*
_output_shapes
: 
Ţ
hidden_layer_3/pool_1/MaxPoolMaxPool'hidden_layer_2/conv_1/LeakyRelu/Maximum*
ksize
*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*
strides
*
data_formatNHWC
m
hidden_layer_3/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:

hidden_layer_3/MeanMeanhidden_layer_3/pool_1/MaxPoolhidden_layer_3/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
r
hidden_layer_3/weights/tagsConst*'
valueB Bhidden_layer_3/weights*
dtype0*
_output_shapes
: 
z
hidden_layer_3/weightsScalarSummaryhidden_layer_3/weights/tagshidden_layer_3/Mean*
_output_shapes
: *
T0
f
!hidden_layer_3/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
"hidden_layer_3/zero_fraction/EqualEqualhidden_layer_3/pool_1/MaxPool!hidden_layer_3/zero_fraction/zero*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

!hidden_layer_3/zero_fraction/CastCast"hidden_layer_3/zero_fraction/Equal*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *

DstT0*

SrcT0

{
"hidden_layer_3/zero_fraction/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Ž
!hidden_layer_3/zero_fraction/MeanMean!hidden_layer_3/zero_fraction/Cast"hidden_layer_3/zero_fraction/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
t
hidden_layer_3/sparsity/tagsConst*(
valueB Bhidden_layer_3/sparsity*
dtype0*
_output_shapes
: 

hidden_layer_3/sparsityScalarSummaryhidden_layer_3/sparsity/tags!hidden_layer_3/zero_fraction/Mean*
T0*
_output_shapes
: 
s
hidden_layer_3/pool_1_1/tagConst*(
valueB Bhidden_layer_3/pool_1_1*
dtype0*
_output_shapes
: 

hidden_layer_3/pool_1_1HistogramSummaryhidden_layer_3/pool_1_1/taghidden_layer_3/pool_1/MaxPool*
_output_shapes
: *
T0
y
hidden_layer_4/flatten/ShapeShapehidden_layer_3/pool_1/MaxPool*
T0*
out_type0*
_output_shapes
:
t
*hidden_layer_4/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
v
,hidden_layer_4/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
v
,hidden_layer_4/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ě
$hidden_layer_4/flatten/strided_sliceStridedSlicehidden_layer_4/flatten/Shape*hidden_layer_4/flatten/strided_slice/stack,hidden_layer_4/flatten/strided_slice/stack_1,hidden_layer_4/flatten/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
q
&hidden_layer_4/flatten/Reshape/shape/1Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
´
$hidden_layer_4/flatten/Reshape/shapePack$hidden_layer_4/flatten/strided_slice&hidden_layer_4/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Ż
hidden_layer_4/flatten/ReshapeReshapehidden_layer_3/pool_1/MaxPool$hidden_layer_4/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
:hidden_layer_5/fcl/kernel/Initializer/random_uniform/shapeConst*
valueB"      *,
_class"
 loc:@hidden_layer_5/fcl/kernel*
dtype0*
_output_shapes
:
Ť
8hidden_layer_5/fcl/kernel/Initializer/random_uniform/minConst*
valueB
 *AW˝*,
_class"
 loc:@hidden_layer_5/fcl/kernel*
dtype0*
_output_shapes
: 
Ť
8hidden_layer_5/fcl/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *AW=*,
_class"
 loc:@hidden_layer_5/fcl/kernel*
dtype0

Bhidden_layer_5/fcl/kernel/Initializer/random_uniform/RandomUniformRandomUniform:hidden_layer_5/fcl/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*,
_class"
 loc:@hidden_layer_5/fcl/kernel*
seed2 

8hidden_layer_5/fcl/kernel/Initializer/random_uniform/subSub8hidden_layer_5/fcl/kernel/Initializer/random_uniform/max8hidden_layer_5/fcl/kernel/Initializer/random_uniform/min*,
_class"
 loc:@hidden_layer_5/fcl/kernel*
_output_shapes
: *
T0

8hidden_layer_5/fcl/kernel/Initializer/random_uniform/mulMulBhidden_layer_5/fcl/kernel/Initializer/random_uniform/RandomUniform8hidden_layer_5/fcl/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@hidden_layer_5/fcl/kernel* 
_output_shapes
:


4hidden_layer_5/fcl/kernel/Initializer/random_uniformAdd8hidden_layer_5/fcl/kernel/Initializer/random_uniform/mul8hidden_layer_5/fcl/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@hidden_layer_5/fcl/kernel* 
_output_shapes
:

ż
hidden_layer_5/fcl/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *,
_class"
 loc:@hidden_layer_5/fcl/kernel*
	container *
shape:

ý
 hidden_layer_5/fcl/kernel/AssignAssignhidden_layer_5/fcl/kernel4hidden_layer_5/fcl/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@hidden_layer_5/fcl/kernel*
validate_shape(* 
_output_shapes
:


hidden_layer_5/fcl/kernel/readIdentityhidden_layer_5/fcl/kernel*
T0*,
_class"
 loc:@hidden_layer_5/fcl/kernel* 
_output_shapes
:

°
9hidden_layer_5/fcl/bias/Initializer/zeros/shape_as_tensorConst*
valueB:**
_class 
loc:@hidden_layer_5/fcl/bias*
dtype0*
_output_shapes
:
 
/hidden_layer_5/fcl/bias/Initializer/zeros/ConstConst*
valueB
 *    **
_class 
loc:@hidden_layer_5/fcl/bias*
dtype0*
_output_shapes
: 

)hidden_layer_5/fcl/bias/Initializer/zerosFill9hidden_layer_5/fcl/bias/Initializer/zeros/shape_as_tensor/hidden_layer_5/fcl/bias/Initializer/zeros/Const*
T0*

index_type0**
_class 
loc:@hidden_layer_5/fcl/bias*
_output_shapes	
:
ą
hidden_layer_5/fcl/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name **
_class 
loc:@hidden_layer_5/fcl/bias
ç
hidden_layer_5/fcl/bias/AssignAssignhidden_layer_5/fcl/bias)hidden_layer_5/fcl/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0**
_class 
loc:@hidden_layer_5/fcl/bias

hidden_layer_5/fcl/bias/readIdentityhidden_layer_5/fcl/bias*
T0**
_class 
loc:@hidden_layer_5/fcl/bias*
_output_shapes	
:
ź
hidden_layer_5/fcl/MatMulMatMulhidden_layer_4/flatten/Reshapehidden_layer_5/fcl/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
¨
hidden_layer_5/fcl/BiasAddBiasAddhidden_layer_5/fcl/MatMulhidden_layer_5/fcl/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
hidden_layer_5/fcl/SigmoidSigmoidhidden_layer_5/fcl/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
hidden_layer_5/ConstConst*
_output_shapes
:*
valueB"       *
dtype0

hidden_layer_5/MeanMeanhidden_layer_5/fcl/Sigmoidhidden_layer_5/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
r
hidden_layer_5/weights/tagsConst*
dtype0*
_output_shapes
: *'
valueB Bhidden_layer_5/weights
z
hidden_layer_5/weightsScalarSummaryhidden_layer_5/weights/tagshidden_layer_5/Mean*
T0*
_output_shapes
: 
f
!hidden_layer_5/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

"hidden_layer_5/zero_fraction/EqualEqualhidden_layer_5/fcl/Sigmoid!hidden_layer_5/zero_fraction/zero*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

!hidden_layer_5/zero_fraction/CastCast"hidden_layer_5/zero_fraction/Equal*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
s
"hidden_layer_5/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Ž
!hidden_layer_5/zero_fraction/MeanMean!hidden_layer_5/zero_fraction/Cast"hidden_layer_5/zero_fraction/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
t
hidden_layer_5/sparsity/tagsConst*(
valueB Bhidden_layer_5/sparsity*
dtype0*
_output_shapes
: 

hidden_layer_5/sparsityScalarSummaryhidden_layer_5/sparsity/tags!hidden_layer_5/zero_fraction/Mean*
T0*
_output_shapes
: 
m
hidden_layer_5/fcl_1/tagConst*%
valueB Bhidden_layer_5/fcl_1*
dtype0*
_output_shapes
: 

hidden_layer_5/fcl_1HistogramSummaryhidden_layer_5/fcl_1/taghidden_layer_5/fcl/Sigmoid*
T0*
_output_shapes
: 
ť
;output_layer/logits/kernel/Initializer/random_uniform/shapeConst*
valueB"      *-
_class#
!loc:@output_layer/logits/kernel*
dtype0*
_output_shapes
:
­
9output_layer/logits/kernel/Initializer/random_uniform/minConst*
valueB
 *mJž*-
_class#
!loc:@output_layer/logits/kernel*
dtype0*
_output_shapes
: 
­
9output_layer/logits/kernel/Initializer/random_uniform/maxConst*
valueB
 *mJ>*-
_class#
!loc:@output_layer/logits/kernel*
dtype0*
_output_shapes
: 

Coutput_layer/logits/kernel/Initializer/random_uniform/RandomUniformRandomUniform;output_layer/logits/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*-
_class#
!loc:@output_layer/logits/kernel*
seed2 

9output_layer/logits/kernel/Initializer/random_uniform/subSub9output_layer/logits/kernel/Initializer/random_uniform/max9output_layer/logits/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*-
_class#
!loc:@output_layer/logits/kernel

9output_layer/logits/kernel/Initializer/random_uniform/mulMulCoutput_layer/logits/kernel/Initializer/random_uniform/RandomUniform9output_layer/logits/kernel/Initializer/random_uniform/sub*
T0*-
_class#
!loc:@output_layer/logits/kernel*
_output_shapes
:	

5output_layer/logits/kernel/Initializer/random_uniformAdd9output_layer/logits/kernel/Initializer/random_uniform/mul9output_layer/logits/kernel/Initializer/random_uniform/min*-
_class#
!loc:@output_layer/logits/kernel*
_output_shapes
:	*
T0
ż
output_layer/logits/kernel
VariableV2*
_output_shapes
:	*
shared_name *-
_class#
!loc:@output_layer/logits/kernel*
	container *
shape:	*
dtype0

!output_layer/logits/kernel/AssignAssignoutput_layer/logits/kernel5output_layer/logits/kernel/Initializer/random_uniform*
T0*-
_class#
!loc:@output_layer/logits/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
 
output_layer/logits/kernel/readIdentityoutput_layer/logits/kernel*
_output_shapes
:	*
T0*-
_class#
!loc:@output_layer/logits/kernel
ą
:output_layer/logits/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*+
_class!
loc:@output_layer/logits/bias*
dtype0*
_output_shapes
:
˘
0output_layer/logits/bias/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@output_layer/logits/bias*
dtype0*
_output_shapes
: 

*output_layer/logits/bias/Initializer/zerosFill:output_layer/logits/bias/Initializer/zeros/shape_as_tensor0output_layer/logits/bias/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@output_layer/logits/bias*
_output_shapes
:
ą
output_layer/logits/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@output_layer/logits/bias*
	container *
shape:
ę
output_layer/logits/bias/AssignAssignoutput_layer/logits/bias*output_layer/logits/bias/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@output_layer/logits/bias*
validate_shape(*
_output_shapes
:

output_layer/logits/bias/readIdentityoutput_layer/logits/bias*
_output_shapes
:*
T0*+
_class!
loc:@output_layer/logits/bias
š
output_layer/logits/MatMulMatMulhidden_layer_5/fcl/Sigmoidoutput_layer/logits/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
Ş
output_layer/logits/BiasAddBiasAddoutput_layer/logits/MatMuloutput_layer/logits/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
output_layer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

output_layer/MeanMeanoutput_layer/logits/BiasAddoutput_layer/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
n
output_layer/weights/tagsConst*
_output_shapes
: *%
valueB Boutput_layer/weights*
dtype0
t
output_layer/weightsScalarSummaryoutput_layer/weights/tagsoutput_layer/Mean*
_output_shapes
: *
T0
d
output_layer/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

 output_layer/zero_fraction/EqualEqualoutput_layer/logits/BiasAddoutput_layer/zero_fraction/zero*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

output_layer/zero_fraction/CastCast output_layer/zero_fraction/Equal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0

q
 output_layer/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
¨
output_layer/zero_fraction/MeanMeanoutput_layer/zero_fraction/Cast output_layer/zero_fraction/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
p
output_layer/sparsity/tagsConst*&
valueB Boutput_layer/sparsity*
dtype0*
_output_shapes
: 

output_layer/sparsityScalarSummaryoutput_layer/sparsity/tagsoutput_layer/zero_fraction/Mean*
T0*
_output_shapes
: 
o
output_layer/logits_1/tagConst*&
valueB Boutput_layer/logits_1*
dtype0*
_output_shapes
: 

output_layer/logits_1HistogramSummaryoutput_layer/logits_1/tagoutput_layer/logits/BiasAdd*
T0*
_output_shapes
: 
a
SoftmaxSoftmaxoutput_layer/logits/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
index_to_string/ConstConst*a
valueXBVBABBBCBDBEBFBGBHBIBJBKBLBMBNBOBPBQBRBSBTBUBVBWBXBYBZ*
dtype0*
_output_shapes
:
V
index_to_string/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
]
index_to_string/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
]
index_to_string/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

index_to_string/rangeRangeindex_to_string/range/startindex_to_string/Sizeindex_to_string/range/delta*

Tidx0*
_output_shapes
:
j
index_to_string/ToInt64Castindex_to_string/range*

SrcT0*
_output_shapes
:*

DstT0	

index_to_stringHashTableV2*
_output_shapes
: *
shared_name *
use_node_name_sharing( *
	key_dtype0	*
	container *
value_dtype0
[
index_to_string/Const_1Const*
valueB	 BUNK*
dtype0*
_output_shapes
: 

index_to_string/table_initInitializeTableV2index_to_stringindex_to_string/ToInt64index_to_string/Const*

Tkey0	*

Tval0
J
TopKV2/kConst*
value	B :*
dtype0*
_output_shapes
: 
v
TopKV2TopKV2SoftmaxTopKV2/k*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
sorted(*
T0
W
CastCastTopKV2:1*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0	

index_to_string_LookupLookupTableFindV2index_to_stringArgMaxindex_to_string/Const_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*	
Tin0	*

Tout0

index_to_string_Lookup_1LookupTableFindV2index_to_stringCastindex_to_string/Const_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*	
Tin0	*

Tout0
W
Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
MaxMaxSoftmaxMax/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
	keep_dims( *
T0
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_a3512738f229402ca9eab7ef306779c2/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Ř
save/SaveV2/tensor_namesConst"/device:CPU:0*ü
valueňBď	Bglobal_stepBhidden_layer_0/conv/biasBhidden_layer_0/conv/kernelBhidden_layer_2/conv_1/biasBhidden_layer_2/conv_1/kernelBhidden_layer_5/fcl/biasBhidden_layer_5/fcl/kernelBoutput_layer/logits/biasBoutput_layer/logits/kernel*
dtype0*
_output_shapes
:	

save/SaveV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
ň
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_stephidden_layer_0/conv/biashidden_layer_0/conv/kernelhidden_layer_2/conv_1/biashidden_layer_2/conv_1/kernelhidden_layer_5/fcl/biashidden_layer_5/fcl/kerneloutput_layer/logits/biasoutput_layer/logits/kernel"/device:CPU:0*
dtypes
2		
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
Ź
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
Ű
save/RestoreV2/tensor_namesConst"/device:CPU:0*ü
valueňBď	Bglobal_stepBhidden_layer_0/conv/biasBhidden_layer_0/conv/kernelBhidden_layer_2/conv_1/biasBhidden_layer_2/conv_1/kernelBhidden_layer_5/fcl/biasBhidden_layer_5/fcl/kernelBoutput_layer/logits/biasBoutput_layer/logits/kernel*
dtype0*
_output_shapes
:	

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
Ç
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2		*8
_output_shapes&
$:::::::::

save/AssignAssignglobal_stepsave/RestoreV2*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(
ž
save/Assign_1Assignhidden_layer_0/conv/biassave/RestoreV2:1*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@hidden_layer_0/conv/bias*
validate_shape(
Î
save/Assign_2Assignhidden_layer_0/conv/kernelsave/RestoreV2:2*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*-
_class#
!loc:@hidden_layer_0/conv/kernel
Â
save/Assign_3Assignhidden_layer_2/conv_1/biassave/RestoreV2:3*
_output_shapes
: *
use_locking(*
T0*-
_class#
!loc:@hidden_layer_2/conv_1/bias*
validate_shape(
Ň
save/Assign_4Assignhidden_layer_2/conv_1/kernelsave/RestoreV2:4*
use_locking(*
T0*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel*
validate_shape(*&
_output_shapes
:  
˝
save/Assign_5Assignhidden_layer_5/fcl/biassave/RestoreV2:5*
use_locking(*
T0**
_class 
loc:@hidden_layer_5/fcl/bias*
validate_shape(*
_output_shapes	
:
Ć
save/Assign_6Assignhidden_layer_5/fcl/kernelsave/RestoreV2:6*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*,
_class"
 loc:@hidden_layer_5/fcl/kernel
ž
save/Assign_7Assignoutput_layer/logits/biassave/RestoreV2:7*
use_locking(*
T0*+
_class!
loc:@output_layer/logits/bias*
validate_shape(*
_output_shapes
:
Ç
save/Assign_8Assignoutput_layer/logits/kernelsave/RestoreV2:8*
use_locking(*
T0*-
_class#
!loc:@output_layer/logits/kernel*
validate_shape(*
_output_shapes
:	
¨
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shard

initNoOp
4
init_all_tablesNoOp^index_to_string/table_init

init_1NoOp
4

group_depsNoOp^init^init_all_tables^init_1
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_e2ef98a3be5140efbb34e59a3094f5ba/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
Ú
save_1/SaveV2/tensor_namesConst"/device:CPU:0*ü
valueňBď	Bglobal_stepBhidden_layer_0/conv/biasBhidden_layer_0/conv/kernelBhidden_layer_2/conv_1/biasBhidden_layer_2/conv_1/kernelBhidden_layer_5/fcl/biasBhidden_layer_5/fcl/kernelBoutput_layer/logits/biasBoutput_layer/logits/kernel*
dtype0*
_output_shapes
:	

save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
ú
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesglobal_stephidden_layer_0/conv/biashidden_layer_0/conv/kernelhidden_layer_2/conv_1/biashidden_layer_2/conv_1/kernelhidden_layer_5/fcl/biashidden_layer_5/fcl/kerneloutput_layer/logits/biasoutput_layer/logits/kernel"/device:CPU:0*
dtypes
2		
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: *
T0
˛
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0
Ý
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*ü
valueňBď	Bglobal_stepBhidden_layer_0/conv/biasBhidden_layer_0/conv/kernelBhidden_layer_2/conv_1/biasBhidden_layer_2/conv_1/kernelBhidden_layer_5/fcl/biasBhidden_layer_5/fcl/kernelBoutput_layer/logits/biasBoutput_layer/logits/kernel*
dtype0

!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
Ď
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		
 
save_1/AssignAssignglobal_stepsave_1/RestoreV2*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
Â
save_1/Assign_1Assignhidden_layer_0/conv/biassave_1/RestoreV2:1*
use_locking(*
T0*+
_class!
loc:@hidden_layer_0/conv/bias*
validate_shape(*
_output_shapes
: 
Ň
save_1/Assign_2Assignhidden_layer_0/conv/kernelsave_1/RestoreV2:2*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*-
_class#
!loc:@hidden_layer_0/conv/kernel
Ć
save_1/Assign_3Assignhidden_layer_2/conv_1/biassave_1/RestoreV2:3*
use_locking(*
T0*-
_class#
!loc:@hidden_layer_2/conv_1/bias*
validate_shape(*
_output_shapes
: 
Ö
save_1/Assign_4Assignhidden_layer_2/conv_1/kernelsave_1/RestoreV2:4*
use_locking(*
T0*/
_class%
#!loc:@hidden_layer_2/conv_1/kernel*
validate_shape(*&
_output_shapes
:  
Á
save_1/Assign_5Assignhidden_layer_5/fcl/biassave_1/RestoreV2:5*
use_locking(*
T0**
_class 
loc:@hidden_layer_5/fcl/bias*
validate_shape(*
_output_shapes	
:
Ę
save_1/Assign_6Assignhidden_layer_5/fcl/kernelsave_1/RestoreV2:6*
use_locking(*
T0*,
_class"
 loc:@hidden_layer_5/fcl/kernel*
validate_shape(* 
_output_shapes
:

Â
save_1/Assign_7Assignoutput_layer/logits/biassave_1/RestoreV2:7*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@output_layer/logits/bias*
validate_shape(
Ë
save_1/Assign_8Assignoutput_layer/logits/kernelsave_1/RestoreV2:8*
use_locking(*
T0*-
_class#
!loc:@output_layer/logits/kernel*
validate_shape(*
_output_shapes
:	
ź
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8" 
global_step

global_step:0"˙
	summariesń
î
images:0
learning_rate:0
hidden_layer_0/weights:0
hidden_layer_0/sparsity:0
hidden_layer_0/conv_1:0
hidden_layer_1/weights:0
hidden_layer_1/sparsity:0
hidden_layer_1/pool_1:0
hidden_layer_2/weights:0
hidden_layer_2/sparsity:0
hidden_layer_2/conv_1_1:0
hidden_layer_3/weights:0
hidden_layer_3/sparsity:0
hidden_layer_3/pool_1_1:0
hidden_layer_5/weights:0
hidden_layer_5/sparsity:0
hidden_layer_5/fcl_1:0
output_layer/weights:0
output_layer/sparsity:0
output_layer/logits_1:0"ß	
trainable_variablesÇ	Ä	

hidden_layer_0/conv/kernel:0!hidden_layer_0/conv/kernel/Assign!hidden_layer_0/conv/kernel/read:027hidden_layer_0/conv/kernel/Initializer/random_uniform:0

hidden_layer_0/conv/bias:0hidden_layer_0/conv/bias/Assignhidden_layer_0/conv/bias/read:02,hidden_layer_0/conv/bias/Initializer/zeros:0
Ľ
hidden_layer_2/conv_1/kernel:0#hidden_layer_2/conv_1/kernel/Assign#hidden_layer_2/conv_1/kernel/read:029hidden_layer_2/conv_1/kernel/Initializer/random_uniform:0

hidden_layer_2/conv_1/bias:0!hidden_layer_2/conv_1/bias/Assign!hidden_layer_2/conv_1/bias/read:02.hidden_layer_2/conv_1/bias/Initializer/zeros:0

hidden_layer_5/fcl/kernel:0 hidden_layer_5/fcl/kernel/Assign hidden_layer_5/fcl/kernel/read:026hidden_layer_5/fcl/kernel/Initializer/random_uniform:0

hidden_layer_5/fcl/bias:0hidden_layer_5/fcl/bias/Assignhidden_layer_5/fcl/bias/read:02+hidden_layer_5/fcl/bias/Initializer/zeros:0

output_layer/logits/kernel:0!output_layer/logits/kernel/Assign!output_layer/logits/kernel/read:027output_layer/logits/kernel/Initializer/random_uniform:0

output_layer/logits/bias:0output_layer/logits/bias/Assignoutput_layer/logits/bias/read:02,output_layer/logits/bias/Initializer/zeros:0"Ż

	variablesĄ


X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0

hidden_layer_0/conv/kernel:0!hidden_layer_0/conv/kernel/Assign!hidden_layer_0/conv/kernel/read:027hidden_layer_0/conv/kernel/Initializer/random_uniform:0

hidden_layer_0/conv/bias:0hidden_layer_0/conv/bias/Assignhidden_layer_0/conv/bias/read:02,hidden_layer_0/conv/bias/Initializer/zeros:0
Ľ
hidden_layer_2/conv_1/kernel:0#hidden_layer_2/conv_1/kernel/Assign#hidden_layer_2/conv_1/kernel/read:029hidden_layer_2/conv_1/kernel/Initializer/random_uniform:0

hidden_layer_2/conv_1/bias:0!hidden_layer_2/conv_1/bias/Assign!hidden_layer_2/conv_1/bias/read:02.hidden_layer_2/conv_1/bias/Initializer/zeros:0

hidden_layer_5/fcl/kernel:0 hidden_layer_5/fcl/kernel/Assign hidden_layer_5/fcl/kernel/read:026hidden_layer_5/fcl/kernel/Initializer/random_uniform:0

hidden_layer_5/fcl/bias:0hidden_layer_5/fcl/bias/Assignhidden_layer_5/fcl/bias/read:02+hidden_layer_5/fcl/bias/Initializer/zeros:0

output_layer/logits/kernel:0!output_layer/logits/kernel/Assign!output_layer/logits/kernel/read:027output_layer/logits/kernel/Initializer/random_uniform:0

output_layer/logits/bias:0output_layer/logits/bias/Assignoutput_layer/logits/bias/read:02,output_layer/logits/bias/Initializer/zeros:0" 
legacy_init_op


group_deps"3
table_initializer

index_to_string/table_init*
classes
1
x,
Placeholder:0˙˙˙˙˙˙˙˙˙  ;
output1
index_to_string_Lookup_1:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict*

confidences
1
x,
Placeholder:0˙˙˙˙˙˙˙˙˙  "
output
Max:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict*
scoresz
1
x,
Placeholder:0˙˙˙˙˙˙˙˙˙  )
output
TopKV2:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict*

prediction
1
x,
Placeholder:0˙˙˙˙˙˙˙˙˙  5
output+
index_to_string_Lookup:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict