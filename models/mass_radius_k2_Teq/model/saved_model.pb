��	
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02v2.6.0-rc2-32-g919f693420e8Ȃ
w
relu_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namerelu_0/kernel
p
!relu_0/kernel/Read/ReadVariableOpReadVariableOprelu_0/kernel*
_output_shapes
:	�*
dtype0
o
relu_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namerelu_0/bias
h
relu_0/bias/Read/ReadVariableOpReadVariableOprelu_0/bias*
_output_shapes	
:�*
dtype0
x
relu_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namerelu_1/kernel
q
!relu_1/kernel/Read/ReadVariableOpReadVariableOprelu_1/kernel* 
_output_shapes
:
��*
dtype0
o
relu_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namerelu_1/bias
h
relu_1/bias/Read/ReadVariableOpReadVariableOprelu_1/bias*
_output_shapes	
:�*
dtype0
x
relu_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namerelu_2/kernel
q
!relu_2/kernel/Read/ReadVariableOpReadVariableOprelu_2/kernel* 
_output_shapes
:
��*
dtype0
o
relu_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namerelu_2/bias
h
relu_2/bias/Read/ReadVariableOpReadVariableOprelu_2/bias*
_output_shapes	
:�*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
output_mdn/mus/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameoutput_mdn/mus/kernel
�
)output_mdn/mus/kernel/Read/ReadVariableOpReadVariableOpoutput_mdn/mus/kernel* 
_output_shapes
:
��*
dtype0

output_mdn/mus/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameoutput_mdn/mus/bias
x
'output_mdn/mus/bias/Read/ReadVariableOpReadVariableOpoutput_mdn/mus/bias*
_output_shapes	
:�*
dtype0
�
output_mdn/sigmas/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameoutput_mdn/sigmas/kernel
�
,output_mdn/sigmas/kernel/Read/ReadVariableOpReadVariableOpoutput_mdn/sigmas/kernel* 
_output_shapes
:
��*
dtype0
�
output_mdn/sigmas/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameoutput_mdn/sigmas/bias
~
*output_mdn/sigmas/bias/Read/ReadVariableOpReadVariableOpoutput_mdn/sigmas/bias*
_output_shapes	
:�*
dtype0
�
output_mdn/pis/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*&
shared_nameoutput_mdn/pis/kernel
�
)output_mdn/pis/kernel/Read/ReadVariableOpReadVariableOpoutput_mdn/pis/kernel*
_output_shapes
:	�2*
dtype0
~
output_mdn/pis/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameoutput_mdn/pis/bias
w
'output_mdn/pis/bias/Read/ReadVariableOpReadVariableOpoutput_mdn/pis/bias*
_output_shapes
:2*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/relu_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/relu_0/kernel/m
~
(Adam/relu_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/relu_0/kernel/m*
_output_shapes
:	�*
dtype0
}
Adam/relu_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/relu_0/bias/m
v
&Adam/relu_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/relu_0/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/relu_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/relu_1/kernel/m

(Adam/relu_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/relu_1/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/relu_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/relu_1/bias/m
v
&Adam/relu_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/relu_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/relu_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/relu_2/kernel/m

(Adam/relu_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/relu_2/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/relu_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/relu_2/bias/m
v
&Adam/relu_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/relu_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/output_mdn/mus/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_nameAdam/output_mdn/mus/kernel/m
�
0Adam/output_mdn/mus/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_mdn/mus/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/output_mdn/mus/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/output_mdn/mus/bias/m
�
.Adam/output_mdn/mus/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_mdn/mus/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/output_mdn/sigmas/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*0
shared_name!Adam/output_mdn/sigmas/kernel/m
�
3Adam/output_mdn/sigmas/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_mdn/sigmas/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/output_mdn/sigmas/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nameAdam/output_mdn/sigmas/bias/m
�
1Adam/output_mdn/sigmas/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_mdn/sigmas/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/output_mdn/pis/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*-
shared_nameAdam/output_mdn/pis/kernel/m
�
0Adam/output_mdn/pis/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_mdn/pis/kernel/m*
_output_shapes
:	�2*
dtype0
�
Adam/output_mdn/pis/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*+
shared_nameAdam/output_mdn/pis/bias/m
�
.Adam/output_mdn/pis/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_mdn/pis/bias/m*
_output_shapes
:2*
dtype0
�
Adam/relu_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/relu_0/kernel/v
~
(Adam/relu_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/relu_0/kernel/v*
_output_shapes
:	�*
dtype0
}
Adam/relu_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/relu_0/bias/v
v
&Adam/relu_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/relu_0/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/relu_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/relu_1/kernel/v

(Adam/relu_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/relu_1/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/relu_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/relu_1/bias/v
v
&Adam/relu_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/relu_1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/relu_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/relu_2/kernel/v

(Adam/relu_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/relu_2/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/relu_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/relu_2/bias/v
v
&Adam/relu_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/relu_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/output_mdn/mus/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_nameAdam/output_mdn/mus/kernel/v
�
0Adam/output_mdn/mus/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_mdn/mus/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/output_mdn/mus/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/output_mdn/mus/bias/v
�
.Adam/output_mdn/mus/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_mdn/mus/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/output_mdn/sigmas/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*0
shared_name!Adam/output_mdn/sigmas/kernel/v
�
3Adam/output_mdn/sigmas/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_mdn/sigmas/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/output_mdn/sigmas/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nameAdam/output_mdn/sigmas/bias/v
�
1Adam/output_mdn/sigmas/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_mdn/sigmas/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/output_mdn/pis/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*-
shared_nameAdam/output_mdn/pis/kernel/v
�
0Adam/output_mdn/pis/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_mdn/pis/kernel/v*
_output_shapes
:	�2*
dtype0
�
Adam/output_mdn/pis/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*+
shared_nameAdam/output_mdn/pis/bias/v
�
.Adam/output_mdn/pis/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_mdn/pis/bias/v*
_output_shapes
:2*
dtype0

NoOpNoOp
�@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�?
value�?B�? B�?
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
{
mdn_mus

mdn_sigmas

mdn_pi
 	variables
!trainable_variables
"regularization_losses
#	keras_api
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemhmimjmkmlmm)mn*mo+mp,mq-mr.msvtvuvvvwvxvy)vz*v{+v|,v}-v~.v
V
0
1
2
3
4
5
)6
*7
+8
,9
-10
.11
V
0
1
2
3
4
5
)6
*7
+8
,9
-10
.11
 
�
/non_trainable_variables
0layer_regularization_losses

1layers
2metrics
3layer_metrics
trainable_variables
	variables
regularization_losses
 
YW
VARIABLE_VALUErelu_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUErelu_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
4non_trainable_variables
5layer_regularization_losses
6layer_metrics
7metrics
	variables
trainable_variables

8layers
regularization_losses
YW
VARIABLE_VALUErelu_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUErelu_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
9non_trainable_variables
:layer_regularization_losses
;layer_metrics
<metrics
	variables
trainable_variables

=layers
regularization_losses
YW
VARIABLE_VALUErelu_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUErelu_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
>non_trainable_variables
?layer_regularization_losses
@layer_metrics
Ametrics
	variables
trainable_variables

Blayers
regularization_losses
h

)kernel
*bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
h

+kernel
,bias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
h

-kernel
.bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
*
)0
*1
+2
,3
-4
.5
*
)0
*1
+2
,3
-4
.5
 
�
Onon_trainable_variables
Player_regularization_losses
Qlayer_metrics
Rmetrics
 	variables
!trainable_variables

Slayers
"regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_mdn/mus/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEoutput_mdn/mus/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEoutput_mdn/sigmas/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEoutput_mdn/sigmas/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEoutput_mdn/pis/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEoutput_mdn/pis/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2
3

T0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

)0
*1

)0
*1
 
�
Unon_trainable_variables
Vlayer_regularization_losses
Wlayer_metrics
Xmetrics
C	variables
Dtrainable_variables

Ylayers
Eregularization_losses

+0
,1

+0
,1
 
�
Znon_trainable_variables
[layer_regularization_losses
\layer_metrics
]metrics
G	variables
Htrainable_variables

^layers
Iregularization_losses

-0
.1

-0
.1
 
�
_non_trainable_variables
`layer_regularization_losses
alayer_metrics
bmetrics
K	variables
Ltrainable_variables

clayers
Mregularization_losses
 
 
 
 

0
1
2
4
	dtotal
	ecount
f	variables
g	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

d0
e1

f	variables
|z
VARIABLE_VALUEAdam/relu_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/relu_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/relu_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/relu_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/relu_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/relu_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output_mdn/mus/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_mdn/mus/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/output_mdn/sigmas/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/output_mdn/sigmas/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/output_mdn/pis/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output_mdn/pis/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/relu_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/relu_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/relu_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/relu_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/relu_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/relu_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output_mdn/mus/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output_mdn/mus/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/output_mdn/sigmas/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/output_mdn/sigmas/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/output_mdn/pis/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output_mdn/pis/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputrelu_0/kernelrelu_0/biasrelu_1/kernelrelu_1/biasrelu_2/kernelrelu_2/biasoutput_mdn/mus/kerneloutput_mdn/mus/biasoutput_mdn/sigmas/kerneloutput_mdn/sigmas/biasoutput_mdn/pis/kerneloutput_mdn/pis/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *-
f(R&
$__inference_signature_wrapper_649938
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!relu_0/kernel/Read/ReadVariableOprelu_0/bias/Read/ReadVariableOp!relu_1/kernel/Read/ReadVariableOprelu_1/bias/Read/ReadVariableOp!relu_2/kernel/Read/ReadVariableOprelu_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)output_mdn/mus/kernel/Read/ReadVariableOp'output_mdn/mus/bias/Read/ReadVariableOp,output_mdn/sigmas/kernel/Read/ReadVariableOp*output_mdn/sigmas/bias/Read/ReadVariableOp)output_mdn/pis/kernel/Read/ReadVariableOp'output_mdn/pis/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/relu_0/kernel/m/Read/ReadVariableOp&Adam/relu_0/bias/m/Read/ReadVariableOp(Adam/relu_1/kernel/m/Read/ReadVariableOp&Adam/relu_1/bias/m/Read/ReadVariableOp(Adam/relu_2/kernel/m/Read/ReadVariableOp&Adam/relu_2/bias/m/Read/ReadVariableOp0Adam/output_mdn/mus/kernel/m/Read/ReadVariableOp.Adam/output_mdn/mus/bias/m/Read/ReadVariableOp3Adam/output_mdn/sigmas/kernel/m/Read/ReadVariableOp1Adam/output_mdn/sigmas/bias/m/Read/ReadVariableOp0Adam/output_mdn/pis/kernel/m/Read/ReadVariableOp.Adam/output_mdn/pis/bias/m/Read/ReadVariableOp(Adam/relu_0/kernel/v/Read/ReadVariableOp&Adam/relu_0/bias/v/Read/ReadVariableOp(Adam/relu_1/kernel/v/Read/ReadVariableOp&Adam/relu_1/bias/v/Read/ReadVariableOp(Adam/relu_2/kernel/v/Read/ReadVariableOp&Adam/relu_2/bias/v/Read/ReadVariableOp0Adam/output_mdn/mus/kernel/v/Read/ReadVariableOp.Adam/output_mdn/mus/bias/v/Read/ReadVariableOp3Adam/output_mdn/sigmas/kernel/v/Read/ReadVariableOp1Adam/output_mdn/sigmas/bias/v/Read/ReadVariableOp0Adam/output_mdn/pis/kernel/v/Read/ReadVariableOp.Adam/output_mdn/pis/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *(
f#R!
__inference__traced_save_650354
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerelu_0/kernelrelu_0/biasrelu_1/kernelrelu_1/biasrelu_2/kernelrelu_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateoutput_mdn/mus/kerneloutput_mdn/mus/biasoutput_mdn/sigmas/kerneloutput_mdn/sigmas/biasoutput_mdn/pis/kerneloutput_mdn/pis/biastotalcountAdam/relu_0/kernel/mAdam/relu_0/bias/mAdam/relu_1/kernel/mAdam/relu_1/bias/mAdam/relu_2/kernel/mAdam/relu_2/bias/mAdam/output_mdn/mus/kernel/mAdam/output_mdn/mus/bias/mAdam/output_mdn/sigmas/kernel/mAdam/output_mdn/sigmas/bias/mAdam/output_mdn/pis/kernel/mAdam/output_mdn/pis/bias/mAdam/relu_0/kernel/vAdam/relu_0/bias/vAdam/relu_1/kernel/vAdam/relu_1/bias/vAdam/relu_2/kernel/vAdam/relu_2/bias/vAdam/output_mdn/mus/kernel/vAdam/output_mdn/mus/bias/vAdam/output_mdn/sigmas/kernel/vAdam/output_mdn/sigmas/bias/vAdam/output_mdn/pis/kernel/vAdam/output_mdn/pis/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *+
f&R$
"__inference__traced_restore_650493��
�
�
$__inference_MDN_layer_call_fn_649837	
input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�2

unknown_10:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *H
fCRA
?__inference_MDN_layer_call_and_return_conditional_losses_6497812
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameinput
�
�
'__inference_relu_1_layer_call_fn_650136

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_1_layer_call_and_return_conditional_losses_6495762
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_relu_1_layer_call_and_return_conditional_losses_650127

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_relu_0_layer_call_and_return_conditional_losses_649559

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
+__inference_output_mdn_layer_call_fn_650202

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�2
	unknown_4:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *O
fJRH
F__inference_output_mdn_layer_call_and_return_conditional_losses_6496282
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_relu_0_layer_call_and_return_conditional_losses_650107

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�[
�
__inference__traced_save_650354
file_prefix,
(savev2_relu_0_kernel_read_readvariableop*
&savev2_relu_0_bias_read_readvariableop,
(savev2_relu_1_kernel_read_readvariableop*
&savev2_relu_1_bias_read_readvariableop,
(savev2_relu_2_kernel_read_readvariableop*
&savev2_relu_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_output_mdn_mus_kernel_read_readvariableop2
.savev2_output_mdn_mus_bias_read_readvariableop7
3savev2_output_mdn_sigmas_kernel_read_readvariableop5
1savev2_output_mdn_sigmas_bias_read_readvariableop4
0savev2_output_mdn_pis_kernel_read_readvariableop2
.savev2_output_mdn_pis_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_relu_0_kernel_m_read_readvariableop1
-savev2_adam_relu_0_bias_m_read_readvariableop3
/savev2_adam_relu_1_kernel_m_read_readvariableop1
-savev2_adam_relu_1_bias_m_read_readvariableop3
/savev2_adam_relu_2_kernel_m_read_readvariableop1
-savev2_adam_relu_2_bias_m_read_readvariableop;
7savev2_adam_output_mdn_mus_kernel_m_read_readvariableop9
5savev2_adam_output_mdn_mus_bias_m_read_readvariableop>
:savev2_adam_output_mdn_sigmas_kernel_m_read_readvariableop<
8savev2_adam_output_mdn_sigmas_bias_m_read_readvariableop;
7savev2_adam_output_mdn_pis_kernel_m_read_readvariableop9
5savev2_adam_output_mdn_pis_bias_m_read_readvariableop3
/savev2_adam_relu_0_kernel_v_read_readvariableop1
-savev2_adam_relu_0_bias_v_read_readvariableop3
/savev2_adam_relu_1_kernel_v_read_readvariableop1
-savev2_adam_relu_1_bias_v_read_readvariableop3
/savev2_adam_relu_2_kernel_v_read_readvariableop1
-savev2_adam_relu_2_bias_v_read_readvariableop;
7savev2_adam_output_mdn_mus_kernel_v_read_readvariableop9
5savev2_adam_output_mdn_mus_bias_v_read_readvariableop>
:savev2_adam_output_mdn_sigmas_kernel_v_read_readvariableop<
8savev2_adam_output_mdn_sigmas_bias_v_read_readvariableop;
7savev2_adam_output_mdn_pis_kernel_v_read_readvariableop9
5savev2_adam_output_mdn_pis_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*�
value�B�,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_relu_0_kernel_read_readvariableop&savev2_relu_0_bias_read_readvariableop(savev2_relu_1_kernel_read_readvariableop&savev2_relu_1_bias_read_readvariableop(savev2_relu_2_kernel_read_readvariableop&savev2_relu_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_output_mdn_mus_kernel_read_readvariableop.savev2_output_mdn_mus_bias_read_readvariableop3savev2_output_mdn_sigmas_kernel_read_readvariableop1savev2_output_mdn_sigmas_bias_read_readvariableop0savev2_output_mdn_pis_kernel_read_readvariableop.savev2_output_mdn_pis_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_relu_0_kernel_m_read_readvariableop-savev2_adam_relu_0_bias_m_read_readvariableop/savev2_adam_relu_1_kernel_m_read_readvariableop-savev2_adam_relu_1_bias_m_read_readvariableop/savev2_adam_relu_2_kernel_m_read_readvariableop-savev2_adam_relu_2_bias_m_read_readvariableop7savev2_adam_output_mdn_mus_kernel_m_read_readvariableop5savev2_adam_output_mdn_mus_bias_m_read_readvariableop:savev2_adam_output_mdn_sigmas_kernel_m_read_readvariableop8savev2_adam_output_mdn_sigmas_bias_m_read_readvariableop7savev2_adam_output_mdn_pis_kernel_m_read_readvariableop5savev2_adam_output_mdn_pis_bias_m_read_readvariableop/savev2_adam_relu_0_kernel_v_read_readvariableop-savev2_adam_relu_0_bias_v_read_readvariableop/savev2_adam_relu_1_kernel_v_read_readvariableop-savev2_adam_relu_1_bias_v_read_readvariableop/savev2_adam_relu_2_kernel_v_read_readvariableop-savev2_adam_relu_2_bias_v_read_readvariableop7savev2_adam_output_mdn_mus_kernel_v_read_readvariableop5savev2_adam_output_mdn_mus_bias_v_read_readvariableop:savev2_adam_output_mdn_sigmas_kernel_v_read_readvariableop8savev2_adam_output_mdn_sigmas_bias_v_read_readvariableop7savev2_adam_output_mdn_pis_kernel_v_read_readvariableop5savev2_adam_output_mdn_pis_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:�:
��:�:
��:�: : : : : :
��:�:
��:�:	�2:2: : :	�:�:
��:�:
��:�:
��:�:
��:�:	�2:2:	�:�:
��:�:
��:�:
��:�:
��:�:	�2:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�2: 

_output_shapes
:2:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�2: 

_output_shapes
:2:% !

_output_shapes
:	�:!!

_output_shapes	
:�:&""
 
_output_shapes
:
��:!#

_output_shapes	
:�:&$"
 
_output_shapes
:
��:!%

_output_shapes	
:�:&&"
 
_output_shapes
:
��:!'

_output_shapes	
:�:&("
 
_output_shapes
:
��:!)

_output_shapes	
:�:%*!

_output_shapes
:	�2: +

_output_shapes
:2:,

_output_shapes
: 
�I
�

?__inference_MDN_layer_call_and_return_conditional_losses_650038

inputs8
%relu_0_matmul_readvariableop_resource:	�5
&relu_0_biasadd_readvariableop_resource:	�9
%relu_1_matmul_readvariableop_resource:
��5
&relu_1_biasadd_readvariableop_resource:	�9
%relu_2_matmul_readvariableop_resource:
��5
&relu_2_biasadd_readvariableop_resource:	�I
5output_mdn_mdn_mdn_mus_matmul_readvariableop_resource:
��E
6output_mdn_mdn_mdn_mus_biasadd_readvariableop_resource:	�L
8output_mdn_mdn_mdn_sigmas_matmul_readvariableop_resource:
��H
9output_mdn_mdn_mdn_sigmas_biasadd_readvariableop_resource:	�G
4output_mdn_mdn_mdn_pi_matmul_readvariableop_resource:	�2C
5output_mdn_mdn_mdn_pi_biasadd_readvariableop_resource:2
identity��-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp�,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp�,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp�+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp�0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp�/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp�relu_0/BiasAdd/ReadVariableOp�relu_0/MatMul/ReadVariableOp�relu_1/BiasAdd/ReadVariableOp�relu_1/MatMul/ReadVariableOp�relu_2/BiasAdd/ReadVariableOp�relu_2/MatMul/ReadVariableOp�
relu_0/MatMul/ReadVariableOpReadVariableOp%relu_0_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
relu_0/MatMul/ReadVariableOp�
relu_0/MatMulMatMulinputs$relu_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_0/MatMul�
relu_0/BiasAdd/ReadVariableOpReadVariableOp&relu_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
relu_0/BiasAdd/ReadVariableOp�
relu_0/BiasAddBiasAddrelu_0/MatMul:product:0%relu_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_0/BiasAddn
relu_0/ReluRelurelu_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
relu_0/Relu�
relu_1/MatMul/ReadVariableOpReadVariableOp%relu_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
relu_1/MatMul/ReadVariableOp�
relu_1/MatMulMatMulrelu_0/Relu:activations:0$relu_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_1/MatMul�
relu_1/BiasAdd/ReadVariableOpReadVariableOp&relu_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
relu_1/BiasAdd/ReadVariableOp�
relu_1/BiasAddBiasAddrelu_1/MatMul:product:0%relu_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_1/BiasAddn
relu_1/ReluRelurelu_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
relu_1/Relu�
relu_2/MatMul/ReadVariableOpReadVariableOp%relu_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
relu_2/MatMul/ReadVariableOp�
relu_2/MatMulMatMulrelu_1/Relu:activations:0$relu_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_2/MatMul�
relu_2/BiasAdd/ReadVariableOpReadVariableOp&relu_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
relu_2/BiasAdd/ReadVariableOp�
relu_2/BiasAddBiasAddrelu_2/MatMul:product:0%relu_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_2/BiasAddn
relu_2/ReluRelurelu_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
relu_2/Relu�
,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOpReadVariableOp5output_mdn_mdn_mdn_mus_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp�
output_mdn/MDN/mdn_mus/MatMulMatMulrelu_2/Relu:activations:04output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
output_mdn/MDN/mdn_mus/MatMul�
-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOpReadVariableOp6output_mdn_mdn_mdn_mus_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp�
output_mdn/MDN/mdn_mus/BiasAddBiasAdd'output_mdn/MDN/mdn_mus/MatMul:product:05output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
output_mdn/MDN/mdn_mus/BiasAdd�
/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOpReadVariableOp8output_mdn_mdn_mdn_sigmas_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype021
/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp�
 output_mdn/MDN/mdn_sigmas/MatMulMatMulrelu_2/Relu:activations:07output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2"
 output_mdn/MDN/mdn_sigmas/MatMul�
0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOpReadVariableOp9output_mdn_mdn_mdn_sigmas_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp�
!output_mdn/MDN/mdn_sigmas/BiasAddBiasAdd*output_mdn/MDN/mdn_sigmas/MatMul:product:08output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!output_mdn/MDN/mdn_sigmas/BiasAdd�
output_mdn/MDN/mdn_sigmas/EluElu*output_mdn/MDN/mdn_sigmas/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
output_mdn/MDN/mdn_sigmas/Elu�
output_mdn/MDN/mdn_sigmas/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
output_mdn/MDN/mdn_sigmas/add/y�
output_mdn/MDN/mdn_sigmas/addAddV2+output_mdn/MDN/mdn_sigmas/Elu:activations:0(output_mdn/MDN/mdn_sigmas/add/y:output:0*
T0*(
_output_shapes
:����������2
output_mdn/MDN/mdn_sigmas/add�
!output_mdn/MDN/mdn_sigmas/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32#
!output_mdn/MDN/mdn_sigmas/add_1/y�
output_mdn/MDN/mdn_sigmas/add_1AddV2!output_mdn/MDN/mdn_sigmas/add:z:0*output_mdn/MDN/mdn_sigmas/add_1/y:output:0*
T0*(
_output_shapes
:����������2!
output_mdn/MDN/mdn_sigmas/add_1�
+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOpReadVariableOp4output_mdn_mdn_mdn_pi_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype02-
+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp�
output_mdn/MDN/mdn_pi/MatMulMatMulrelu_2/Relu:activations:03output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
output_mdn/MDN/mdn_pi/MatMul�
,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOpReadVariableOp5output_mdn_mdn_mdn_pi_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02.
,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp�
output_mdn/MDN/mdn_pi/BiasAddBiasAdd&output_mdn/MDN/mdn_pi/MatMul:product:04output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
output_mdn/MDN/mdn_pi/BiasAdd�
&output_mdn/MDN/mdn_outputs/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&output_mdn/MDN/mdn_outputs/concat/axis�
!output_mdn/MDN/mdn_outputs/concatConcatV2'output_mdn/MDN/mdn_mus/BiasAdd:output:0#output_mdn/MDN/mdn_sigmas/add_1:z:0&output_mdn/MDN/mdn_pi/BiasAdd:output:0/output_mdn/MDN/mdn_outputs/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2#
!output_mdn/MDN/mdn_outputs/concat�
IdentityIdentity*output_mdn/MDN/mdn_outputs/concat:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp.^output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp-^output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp-^output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp,^output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp1^output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp0^output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp^relu_0/BiasAdd/ReadVariableOp^relu_0/MatMul/ReadVariableOp^relu_1/BiasAdd/ReadVariableOp^relu_1/MatMul/ReadVariableOp^relu_2/BiasAdd/ReadVariableOp^relu_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 2^
-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp2\
,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp2\
,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp2Z
+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp2d
0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp2b
/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp2>
relu_0/BiasAdd/ReadVariableOprelu_0/BiasAdd/ReadVariableOp2<
relu_0/MatMul/ReadVariableOprelu_0/MatMul/ReadVariableOp2>
relu_1/BiasAdd/ReadVariableOprelu_1/BiasAdd/ReadVariableOp2<
relu_1/MatMul/ReadVariableOprelu_1/MatMul/ReadVariableOp2>
relu_2/BiasAdd/ReadVariableOprelu_2/BiasAdd/ReadVariableOp2<
relu_2/MatMul/ReadVariableOprelu_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_MDN_layer_call_fn_650067

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�2

unknown_10:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *H
fCRA
?__inference_MDN_layer_call_and_return_conditional_losses_6496432
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_MDN_layer_call_fn_650096

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�2

unknown_10:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *H
fCRA
?__inference_MDN_layer_call_and_return_conditional_losses_6497812
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�I
�

?__inference_MDN_layer_call_and_return_conditional_losses_649988

inputs8
%relu_0_matmul_readvariableop_resource:	�5
&relu_0_biasadd_readvariableop_resource:	�9
%relu_1_matmul_readvariableop_resource:
��5
&relu_1_biasadd_readvariableop_resource:	�9
%relu_2_matmul_readvariableop_resource:
��5
&relu_2_biasadd_readvariableop_resource:	�I
5output_mdn_mdn_mdn_mus_matmul_readvariableop_resource:
��E
6output_mdn_mdn_mdn_mus_biasadd_readvariableop_resource:	�L
8output_mdn_mdn_mdn_sigmas_matmul_readvariableop_resource:
��H
9output_mdn_mdn_mdn_sigmas_biasadd_readvariableop_resource:	�G
4output_mdn_mdn_mdn_pi_matmul_readvariableop_resource:	�2C
5output_mdn_mdn_mdn_pi_biasadd_readvariableop_resource:2
identity��-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp�,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp�,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp�+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp�0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp�/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp�relu_0/BiasAdd/ReadVariableOp�relu_0/MatMul/ReadVariableOp�relu_1/BiasAdd/ReadVariableOp�relu_1/MatMul/ReadVariableOp�relu_2/BiasAdd/ReadVariableOp�relu_2/MatMul/ReadVariableOp�
relu_0/MatMul/ReadVariableOpReadVariableOp%relu_0_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
relu_0/MatMul/ReadVariableOp�
relu_0/MatMulMatMulinputs$relu_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_0/MatMul�
relu_0/BiasAdd/ReadVariableOpReadVariableOp&relu_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
relu_0/BiasAdd/ReadVariableOp�
relu_0/BiasAddBiasAddrelu_0/MatMul:product:0%relu_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_0/BiasAddn
relu_0/ReluRelurelu_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
relu_0/Relu�
relu_1/MatMul/ReadVariableOpReadVariableOp%relu_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
relu_1/MatMul/ReadVariableOp�
relu_1/MatMulMatMulrelu_0/Relu:activations:0$relu_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_1/MatMul�
relu_1/BiasAdd/ReadVariableOpReadVariableOp&relu_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
relu_1/BiasAdd/ReadVariableOp�
relu_1/BiasAddBiasAddrelu_1/MatMul:product:0%relu_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_1/BiasAddn
relu_1/ReluRelurelu_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
relu_1/Relu�
relu_2/MatMul/ReadVariableOpReadVariableOp%relu_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
relu_2/MatMul/ReadVariableOp�
relu_2/MatMulMatMulrelu_1/Relu:activations:0$relu_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_2/MatMul�
relu_2/BiasAdd/ReadVariableOpReadVariableOp&relu_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
relu_2/BiasAdd/ReadVariableOp�
relu_2/BiasAddBiasAddrelu_2/MatMul:product:0%relu_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
relu_2/BiasAddn
relu_2/ReluRelurelu_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
relu_2/Relu�
,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOpReadVariableOp5output_mdn_mdn_mdn_mus_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp�
output_mdn/MDN/mdn_mus/MatMulMatMulrelu_2/Relu:activations:04output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
output_mdn/MDN/mdn_mus/MatMul�
-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOpReadVariableOp6output_mdn_mdn_mdn_mus_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp�
output_mdn/MDN/mdn_mus/BiasAddBiasAdd'output_mdn/MDN/mdn_mus/MatMul:product:05output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
output_mdn/MDN/mdn_mus/BiasAdd�
/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOpReadVariableOp8output_mdn_mdn_mdn_sigmas_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype021
/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp�
 output_mdn/MDN/mdn_sigmas/MatMulMatMulrelu_2/Relu:activations:07output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2"
 output_mdn/MDN/mdn_sigmas/MatMul�
0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOpReadVariableOp9output_mdn_mdn_mdn_sigmas_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp�
!output_mdn/MDN/mdn_sigmas/BiasAddBiasAdd*output_mdn/MDN/mdn_sigmas/MatMul:product:08output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!output_mdn/MDN/mdn_sigmas/BiasAdd�
output_mdn/MDN/mdn_sigmas/EluElu*output_mdn/MDN/mdn_sigmas/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
output_mdn/MDN/mdn_sigmas/Elu�
output_mdn/MDN/mdn_sigmas/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
output_mdn/MDN/mdn_sigmas/add/y�
output_mdn/MDN/mdn_sigmas/addAddV2+output_mdn/MDN/mdn_sigmas/Elu:activations:0(output_mdn/MDN/mdn_sigmas/add/y:output:0*
T0*(
_output_shapes
:����������2
output_mdn/MDN/mdn_sigmas/add�
!output_mdn/MDN/mdn_sigmas/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32#
!output_mdn/MDN/mdn_sigmas/add_1/y�
output_mdn/MDN/mdn_sigmas/add_1AddV2!output_mdn/MDN/mdn_sigmas/add:z:0*output_mdn/MDN/mdn_sigmas/add_1/y:output:0*
T0*(
_output_shapes
:����������2!
output_mdn/MDN/mdn_sigmas/add_1�
+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOpReadVariableOp4output_mdn_mdn_mdn_pi_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype02-
+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp�
output_mdn/MDN/mdn_pi/MatMulMatMulrelu_2/Relu:activations:03output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
output_mdn/MDN/mdn_pi/MatMul�
,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOpReadVariableOp5output_mdn_mdn_mdn_pi_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02.
,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp�
output_mdn/MDN/mdn_pi/BiasAddBiasAdd&output_mdn/MDN/mdn_pi/MatMul:product:04output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
output_mdn/MDN/mdn_pi/BiasAdd�
&output_mdn/MDN/mdn_outputs/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&output_mdn/MDN/mdn_outputs/concat/axis�
!output_mdn/MDN/mdn_outputs/concatConcatV2'output_mdn/MDN/mdn_mus/BiasAdd:output:0#output_mdn/MDN/mdn_sigmas/add_1:z:0&output_mdn/MDN/mdn_pi/BiasAdd:output:0/output_mdn/MDN/mdn_outputs/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2#
!output_mdn/MDN/mdn_outputs/concat�
IdentityIdentity*output_mdn/MDN/mdn_outputs/concat:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp.^output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp-^output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp-^output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp,^output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp1^output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp0^output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp^relu_0/BiasAdd/ReadVariableOp^relu_0/MatMul/ReadVariableOp^relu_1/BiasAdd/ReadVariableOp^relu_1/MatMul/ReadVariableOp^relu_2/BiasAdd/ReadVariableOp^relu_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 2^
-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp-output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp2\
,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp,output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp2\
,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp,output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp2Z
+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp+output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp2d
0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp0output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp2b
/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp2>
relu_0/BiasAdd/ReadVariableOprelu_0/BiasAdd/ReadVariableOp2<
relu_0/MatMul/ReadVariableOprelu_0/MatMul/ReadVariableOp2>
relu_1/BiasAdd/ReadVariableOprelu_1/BiasAdd/ReadVariableOp2<
relu_1/MatMul/ReadVariableOprelu_1/MatMul/ReadVariableOp2>
relu_2/BiasAdd/ReadVariableOprelu_2/BiasAdd/ReadVariableOp2<
relu_2/MatMul/ReadVariableOprelu_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_649938	
input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�2

unknown_10:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� **
f%R#
!__inference__wrapped_model_6495412
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameinput
�
�
?__inference_MDN_layer_call_and_return_conditional_losses_649781

inputs 
relu_0_649752:	�
relu_0_649754:	�!
relu_1_649757:
��
relu_1_649759:	�!
relu_2_649762:
��
relu_2_649764:	�%
output_mdn_649767:
�� 
output_mdn_649769:	�%
output_mdn_649771:
�� 
output_mdn_649773:	�$
output_mdn_649775:	�2
output_mdn_649777:2
identity��"output_mdn/StatefulPartitionedCall�relu_0/StatefulPartitionedCall�relu_1/StatefulPartitionedCall�relu_2/StatefulPartitionedCall�
relu_0/StatefulPartitionedCallStatefulPartitionedCallinputsrelu_0_649752relu_0_649754*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_0_layer_call_and_return_conditional_losses_6495592 
relu_0/StatefulPartitionedCall�
relu_1/StatefulPartitionedCallStatefulPartitionedCall'relu_0/StatefulPartitionedCall:output:0relu_1_649757relu_1_649759*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_1_layer_call_and_return_conditional_losses_6495762 
relu_1/StatefulPartitionedCall�
relu_2/StatefulPartitionedCallStatefulPartitionedCall'relu_1/StatefulPartitionedCall:output:0relu_2_649762relu_2_649764*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_2_layer_call_and_return_conditional_losses_6495932 
relu_2/StatefulPartitionedCall�
"output_mdn/StatefulPartitionedCallStatefulPartitionedCall'relu_2/StatefulPartitionedCall:output:0output_mdn_649767output_mdn_649769output_mdn_649771output_mdn_649773output_mdn_649775output_mdn_649777*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *O
fJRH
F__inference_output_mdn_layer_call_and_return_conditional_losses_6496282$
"output_mdn/StatefulPartitionedCall�
IdentityIdentity+output_mdn/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp#^output_mdn/StatefulPartitionedCall^relu_0/StatefulPartitionedCall^relu_1/StatefulPartitionedCall^relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 2H
"output_mdn/StatefulPartitionedCall"output_mdn/StatefulPartitionedCall2@
relu_0/StatefulPartitionedCallrelu_0/StatefulPartitionedCall2@
relu_1/StatefulPartitionedCallrelu_1/StatefulPartitionedCall2@
relu_2/StatefulPartitionedCallrelu_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�N
�
!__inference__wrapped_model_649541	
input<
)mdn_relu_0_matmul_readvariableop_resource:	�9
*mdn_relu_0_biasadd_readvariableop_resource:	�=
)mdn_relu_1_matmul_readvariableop_resource:
��9
*mdn_relu_1_biasadd_readvariableop_resource:	�=
)mdn_relu_2_matmul_readvariableop_resource:
��9
*mdn_relu_2_biasadd_readvariableop_resource:	�M
9mdn_output_mdn_mdn_mdn_mus_matmul_readvariableop_resource:
��I
:mdn_output_mdn_mdn_mdn_mus_biasadd_readvariableop_resource:	�P
<mdn_output_mdn_mdn_mdn_sigmas_matmul_readvariableop_resource:
��L
=mdn_output_mdn_mdn_mdn_sigmas_biasadd_readvariableop_resource:	�K
8mdn_output_mdn_mdn_mdn_pi_matmul_readvariableop_resource:	�2G
9mdn_output_mdn_mdn_mdn_pi_biasadd_readvariableop_resource:2
identity��1MDN/output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp�0MDN/output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp�0MDN/output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp�/MDN/output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp�4MDN/output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp�3MDN/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp�!MDN/relu_0/BiasAdd/ReadVariableOp� MDN/relu_0/MatMul/ReadVariableOp�!MDN/relu_1/BiasAdd/ReadVariableOp� MDN/relu_1/MatMul/ReadVariableOp�!MDN/relu_2/BiasAdd/ReadVariableOp� MDN/relu_2/MatMul/ReadVariableOp�
 MDN/relu_0/MatMul/ReadVariableOpReadVariableOp)mdn_relu_0_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02"
 MDN/relu_0/MatMul/ReadVariableOp�
MDN/relu_0/MatMulMatMulinput(MDN/relu_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/relu_0/MatMul�
!MDN/relu_0/BiasAdd/ReadVariableOpReadVariableOp*mdn_relu_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!MDN/relu_0/BiasAdd/ReadVariableOp�
MDN/relu_0/BiasAddBiasAddMDN/relu_0/MatMul:product:0)MDN/relu_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/relu_0/BiasAddz
MDN/relu_0/ReluReluMDN/relu_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
MDN/relu_0/Relu�
 MDN/relu_1/MatMul/ReadVariableOpReadVariableOp)mdn_relu_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 MDN/relu_1/MatMul/ReadVariableOp�
MDN/relu_1/MatMulMatMulMDN/relu_0/Relu:activations:0(MDN/relu_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/relu_1/MatMul�
!MDN/relu_1/BiasAdd/ReadVariableOpReadVariableOp*mdn_relu_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!MDN/relu_1/BiasAdd/ReadVariableOp�
MDN/relu_1/BiasAddBiasAddMDN/relu_1/MatMul:product:0)MDN/relu_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/relu_1/BiasAddz
MDN/relu_1/ReluReluMDN/relu_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
MDN/relu_1/Relu�
 MDN/relu_2/MatMul/ReadVariableOpReadVariableOp)mdn_relu_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 MDN/relu_2/MatMul/ReadVariableOp�
MDN/relu_2/MatMulMatMulMDN/relu_1/Relu:activations:0(MDN/relu_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/relu_2/MatMul�
!MDN/relu_2/BiasAdd/ReadVariableOpReadVariableOp*mdn_relu_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!MDN/relu_2/BiasAdd/ReadVariableOp�
MDN/relu_2/BiasAddBiasAddMDN/relu_2/MatMul:product:0)MDN/relu_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/relu_2/BiasAddz
MDN/relu_2/ReluReluMDN/relu_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
MDN/relu_2/Relu�
0MDN/output_mdn/MDN/mdn_mus/MatMul/ReadVariableOpReadVariableOp9mdn_output_mdn_mdn_mdn_mus_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype022
0MDN/output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp�
!MDN/output_mdn/MDN/mdn_mus/MatMulMatMulMDN/relu_2/Relu:activations:08MDN/output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!MDN/output_mdn/MDN/mdn_mus/MatMul�
1MDN/output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOpReadVariableOp:mdn_output_mdn_mdn_mdn_mus_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype023
1MDN/output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp�
"MDN/output_mdn/MDN/mdn_mus/BiasAddBiasAdd+MDN/output_mdn/MDN/mdn_mus/MatMul:product:09MDN/output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"MDN/output_mdn/MDN/mdn_mus/BiasAdd�
3MDN/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOpReadVariableOp<mdn_output_mdn_mdn_mdn_sigmas_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype025
3MDN/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp�
$MDN/output_mdn/MDN/mdn_sigmas/MatMulMatMulMDN/relu_2/Relu:activations:0;MDN/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$MDN/output_mdn/MDN/mdn_sigmas/MatMul�
4MDN/output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOpReadVariableOp=mdn_output_mdn_mdn_mdn_sigmas_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4MDN/output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp�
%MDN/output_mdn/MDN/mdn_sigmas/BiasAddBiasAdd.MDN/output_mdn/MDN/mdn_sigmas/MatMul:product:0<MDN/output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%MDN/output_mdn/MDN/mdn_sigmas/BiasAdd�
!MDN/output_mdn/MDN/mdn_sigmas/EluElu.MDN/output_mdn/MDN/mdn_sigmas/BiasAdd:output:0*
T0*(
_output_shapes
:����������2#
!MDN/output_mdn/MDN/mdn_sigmas/Elu�
#MDN/output_mdn/MDN/mdn_sigmas/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2%
#MDN/output_mdn/MDN/mdn_sigmas/add/y�
!MDN/output_mdn/MDN/mdn_sigmas/addAddV2/MDN/output_mdn/MDN/mdn_sigmas/Elu:activations:0,MDN/output_mdn/MDN/mdn_sigmas/add/y:output:0*
T0*(
_output_shapes
:����������2#
!MDN/output_mdn/MDN/mdn_sigmas/add�
%MDN/output_mdn/MDN/mdn_sigmas/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32'
%MDN/output_mdn/MDN/mdn_sigmas/add_1/y�
#MDN/output_mdn/MDN/mdn_sigmas/add_1AddV2%MDN/output_mdn/MDN/mdn_sigmas/add:z:0.MDN/output_mdn/MDN/mdn_sigmas/add_1/y:output:0*
T0*(
_output_shapes
:����������2%
#MDN/output_mdn/MDN/mdn_sigmas/add_1�
/MDN/output_mdn/MDN/mdn_pi/MatMul/ReadVariableOpReadVariableOp8mdn_output_mdn_mdn_mdn_pi_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype021
/MDN/output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp�
 MDN/output_mdn/MDN/mdn_pi/MatMulMatMulMDN/relu_2/Relu:activations:07MDN/output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22"
 MDN/output_mdn/MDN/mdn_pi/MatMul�
0MDN/output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOpReadVariableOp9mdn_output_mdn_mdn_mdn_pi_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype022
0MDN/output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp�
!MDN/output_mdn/MDN/mdn_pi/BiasAddBiasAdd*MDN/output_mdn/MDN/mdn_pi/MatMul:product:08MDN/output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22#
!MDN/output_mdn/MDN/mdn_pi/BiasAdd�
*MDN/output_mdn/MDN/mdn_outputs/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*MDN/output_mdn/MDN/mdn_outputs/concat/axis�
%MDN/output_mdn/MDN/mdn_outputs/concatConcatV2+MDN/output_mdn/MDN/mdn_mus/BiasAdd:output:0'MDN/output_mdn/MDN/mdn_sigmas/add_1:z:0*MDN/output_mdn/MDN/mdn_pi/BiasAdd:output:03MDN/output_mdn/MDN/mdn_outputs/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2'
%MDN/output_mdn/MDN/mdn_outputs/concat�
IdentityIdentity.MDN/output_mdn/MDN/mdn_outputs/concat:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp2^MDN/output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp1^MDN/output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp1^MDN/output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp0^MDN/output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp5^MDN/output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp4^MDN/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp"^MDN/relu_0/BiasAdd/ReadVariableOp!^MDN/relu_0/MatMul/ReadVariableOp"^MDN/relu_1/BiasAdd/ReadVariableOp!^MDN/relu_1/MatMul/ReadVariableOp"^MDN/relu_2/BiasAdd/ReadVariableOp!^MDN/relu_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 2f
1MDN/output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp1MDN/output_mdn/MDN/mdn_mus/BiasAdd/ReadVariableOp2d
0MDN/output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp0MDN/output_mdn/MDN/mdn_mus/MatMul/ReadVariableOp2d
0MDN/output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp0MDN/output_mdn/MDN/mdn_pi/BiasAdd/ReadVariableOp2b
/MDN/output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp/MDN/output_mdn/MDN/mdn_pi/MatMul/ReadVariableOp2l
4MDN/output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp4MDN/output_mdn/MDN/mdn_sigmas/BiasAdd/ReadVariableOp2j
3MDN/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp3MDN/output_mdn/MDN/mdn_sigmas/MatMul/ReadVariableOp2F
!MDN/relu_0/BiasAdd/ReadVariableOp!MDN/relu_0/BiasAdd/ReadVariableOp2D
 MDN/relu_0/MatMul/ReadVariableOp MDN/relu_0/MatMul/ReadVariableOp2F
!MDN/relu_1/BiasAdd/ReadVariableOp!MDN/relu_1/BiasAdd/ReadVariableOp2D
 MDN/relu_1/MatMul/ReadVariableOp MDN/relu_1/MatMul/ReadVariableOp2F
!MDN/relu_2/BiasAdd/ReadVariableOp!MDN/relu_2/BiasAdd/ReadVariableOp2D
 MDN/relu_2/MatMul/ReadVariableOp MDN/relu_2/MatMul/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_nameinput
�
�
?__inference_MDN_layer_call_and_return_conditional_losses_649869	
input 
relu_0_649840:	�
relu_0_649842:	�!
relu_1_649845:
��
relu_1_649847:	�!
relu_2_649850:
��
relu_2_649852:	�%
output_mdn_649855:
�� 
output_mdn_649857:	�%
output_mdn_649859:
�� 
output_mdn_649861:	�$
output_mdn_649863:	�2
output_mdn_649865:2
identity��"output_mdn/StatefulPartitionedCall�relu_0/StatefulPartitionedCall�relu_1/StatefulPartitionedCall�relu_2/StatefulPartitionedCall�
relu_0/StatefulPartitionedCallStatefulPartitionedCallinputrelu_0_649840relu_0_649842*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_0_layer_call_and_return_conditional_losses_6495592 
relu_0/StatefulPartitionedCall�
relu_1/StatefulPartitionedCallStatefulPartitionedCall'relu_0/StatefulPartitionedCall:output:0relu_1_649845relu_1_649847*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_1_layer_call_and_return_conditional_losses_6495762 
relu_1/StatefulPartitionedCall�
relu_2/StatefulPartitionedCallStatefulPartitionedCall'relu_1/StatefulPartitionedCall:output:0relu_2_649850relu_2_649852*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_2_layer_call_and_return_conditional_losses_6495932 
relu_2/StatefulPartitionedCall�
"output_mdn/StatefulPartitionedCallStatefulPartitionedCall'relu_2/StatefulPartitionedCall:output:0output_mdn_649855output_mdn_649857output_mdn_649859output_mdn_649861output_mdn_649863output_mdn_649865*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *O
fJRH
F__inference_output_mdn_layer_call_and_return_conditional_losses_6496282$
"output_mdn/StatefulPartitionedCall�
IdentityIdentity+output_mdn/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp#^output_mdn/StatefulPartitionedCall^relu_0/StatefulPartitionedCall^relu_1/StatefulPartitionedCall^relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 2H
"output_mdn/StatefulPartitionedCall"output_mdn/StatefulPartitionedCall2@
relu_0/StatefulPartitionedCallrelu_0/StatefulPartitionedCall2@
relu_1/StatefulPartitionedCallrelu_1/StatefulPartitionedCall2@
relu_2/StatefulPartitionedCallrelu_2/StatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameinput
�
�
B__inference_relu_2_layer_call_and_return_conditional_losses_649593

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
ɺ
�
"__inference__traced_restore_650493
file_prefix1
assignvariableop_relu_0_kernel:	�-
assignvariableop_1_relu_0_bias:	�4
 assignvariableop_2_relu_1_kernel:
��-
assignvariableop_3_relu_1_bias:	�4
 assignvariableop_4_relu_2_kernel:
��-
assignvariableop_5_relu_2_bias:	�&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: =
)assignvariableop_11_output_mdn_mus_kernel:
��6
'assignvariableop_12_output_mdn_mus_bias:	�@
,assignvariableop_13_output_mdn_sigmas_kernel:
��9
*assignvariableop_14_output_mdn_sigmas_bias:	�<
)assignvariableop_15_output_mdn_pis_kernel:	�25
'assignvariableop_16_output_mdn_pis_bias:2#
assignvariableop_17_total: #
assignvariableop_18_count: ;
(assignvariableop_19_adam_relu_0_kernel_m:	�5
&assignvariableop_20_adam_relu_0_bias_m:	�<
(assignvariableop_21_adam_relu_1_kernel_m:
��5
&assignvariableop_22_adam_relu_1_bias_m:	�<
(assignvariableop_23_adam_relu_2_kernel_m:
��5
&assignvariableop_24_adam_relu_2_bias_m:	�D
0assignvariableop_25_adam_output_mdn_mus_kernel_m:
��=
.assignvariableop_26_adam_output_mdn_mus_bias_m:	�G
3assignvariableop_27_adam_output_mdn_sigmas_kernel_m:
��@
1assignvariableop_28_adam_output_mdn_sigmas_bias_m:	�C
0assignvariableop_29_adam_output_mdn_pis_kernel_m:	�2<
.assignvariableop_30_adam_output_mdn_pis_bias_m:2;
(assignvariableop_31_adam_relu_0_kernel_v:	�5
&assignvariableop_32_adam_relu_0_bias_v:	�<
(assignvariableop_33_adam_relu_1_kernel_v:
��5
&assignvariableop_34_adam_relu_1_bias_v:	�<
(assignvariableop_35_adam_relu_2_kernel_v:
��5
&assignvariableop_36_adam_relu_2_bias_v:	�D
0assignvariableop_37_adam_output_mdn_mus_kernel_v:
��=
.assignvariableop_38_adam_output_mdn_mus_bias_v:	�G
3assignvariableop_39_adam_output_mdn_sigmas_kernel_v:
��@
1assignvariableop_40_adam_output_mdn_sigmas_bias_v:	�C
0assignvariableop_41_adam_output_mdn_pis_kernel_v:	�2<
.assignvariableop_42_adam_output_mdn_pis_bias_v:2
identity_44��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*�
value�B�,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_relu_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_relu_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_relu_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_relu_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_relu_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_relu_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_output_mdn_mus_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_output_mdn_mus_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp,assignvariableop_13_output_mdn_sigmas_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_output_mdn_sigmas_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_output_mdn_pis_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_output_mdn_pis_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_relu_0_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_relu_0_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_relu_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_relu_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_relu_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_relu_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_output_mdn_mus_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_output_mdn_mus_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_output_mdn_sigmas_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_output_mdn_sigmas_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_output_mdn_pis_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_output_mdn_pis_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_relu_0_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_relu_0_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_relu_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_relu_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_relu_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_relu_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp0assignvariableop_37_adam_output_mdn_mus_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp.assignvariableop_38_adam_output_mdn_mus_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp3assignvariableop_39_adam_output_mdn_sigmas_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp1assignvariableop_40_adam_output_mdn_sigmas_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp0assignvariableop_41_adam_output_mdn_pis_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp.assignvariableop_42_adam_output_mdn_pis_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43f
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_44�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
B__inference_relu_1_layer_call_and_return_conditional_losses_649576

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_MDN_layer_call_fn_649670	
input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�2

unknown_10:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *H
fCRA
?__inference_MDN_layer_call_and_return_conditional_losses_6496432
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameinput
�
�
?__inference_MDN_layer_call_and_return_conditional_losses_649901	
input 
relu_0_649872:	�
relu_0_649874:	�!
relu_1_649877:
��
relu_1_649879:	�!
relu_2_649882:
��
relu_2_649884:	�%
output_mdn_649887:
�� 
output_mdn_649889:	�%
output_mdn_649891:
�� 
output_mdn_649893:	�$
output_mdn_649895:	�2
output_mdn_649897:2
identity��"output_mdn/StatefulPartitionedCall�relu_0/StatefulPartitionedCall�relu_1/StatefulPartitionedCall�relu_2/StatefulPartitionedCall�
relu_0/StatefulPartitionedCallStatefulPartitionedCallinputrelu_0_649872relu_0_649874*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_0_layer_call_and_return_conditional_losses_6495592 
relu_0/StatefulPartitionedCall�
relu_1/StatefulPartitionedCallStatefulPartitionedCall'relu_0/StatefulPartitionedCall:output:0relu_1_649877relu_1_649879*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_1_layer_call_and_return_conditional_losses_6495762 
relu_1/StatefulPartitionedCall�
relu_2/StatefulPartitionedCallStatefulPartitionedCall'relu_1/StatefulPartitionedCall:output:0relu_2_649882relu_2_649884*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_2_layer_call_and_return_conditional_losses_6495932 
relu_2/StatefulPartitionedCall�
"output_mdn/StatefulPartitionedCallStatefulPartitionedCall'relu_2/StatefulPartitionedCall:output:0output_mdn_649887output_mdn_649889output_mdn_649891output_mdn_649893output_mdn_649895output_mdn_649897*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *O
fJRH
F__inference_output_mdn_layer_call_and_return_conditional_losses_6496282$
"output_mdn/StatefulPartitionedCall�
IdentityIdentity+output_mdn/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp#^output_mdn/StatefulPartitionedCall^relu_0/StatefulPartitionedCall^relu_1/StatefulPartitionedCall^relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 2H
"output_mdn/StatefulPartitionedCall"output_mdn/StatefulPartitionedCall2@
relu_0/StatefulPartitionedCallrelu_0/StatefulPartitionedCall2@
relu_1/StatefulPartitionedCallrelu_1/StatefulPartitionedCall2@
relu_2/StatefulPartitionedCallrelu_2/StatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameinput
�
�
B__inference_relu_2_layer_call_and_return_conditional_losses_650147

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
F__inference_output_mdn_layer_call_and_return_conditional_losses_650185

inputs>
*mdn_mdn_mus_matmul_readvariableop_resource:
��:
+mdn_mdn_mus_biasadd_readvariableop_resource:	�A
-mdn_mdn_sigmas_matmul_readvariableop_resource:
��=
.mdn_mdn_sigmas_biasadd_readvariableop_resource:	�<
)mdn_mdn_pi_matmul_readvariableop_resource:	�28
*mdn_mdn_pi_biasadd_readvariableop_resource:2
identity��"MDN/mdn_mus/BiasAdd/ReadVariableOp�!MDN/mdn_mus/MatMul/ReadVariableOp�!MDN/mdn_pi/BiasAdd/ReadVariableOp� MDN/mdn_pi/MatMul/ReadVariableOp�%MDN/mdn_sigmas/BiasAdd/ReadVariableOp�$MDN/mdn_sigmas/MatMul/ReadVariableOp�
!MDN/mdn_mus/MatMul/ReadVariableOpReadVariableOp*mdn_mdn_mus_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!MDN/mdn_mus/MatMul/ReadVariableOp�
MDN/mdn_mus/MatMulMatMulinputs)MDN/mdn_mus/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/mdn_mus/MatMul�
"MDN/mdn_mus/BiasAdd/ReadVariableOpReadVariableOp+mdn_mdn_mus_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"MDN/mdn_mus/BiasAdd/ReadVariableOp�
MDN/mdn_mus/BiasAddBiasAddMDN/mdn_mus/MatMul:product:0*MDN/mdn_mus/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/mdn_mus/BiasAdd�
$MDN/mdn_sigmas/MatMul/ReadVariableOpReadVariableOp-mdn_mdn_sigmas_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02&
$MDN/mdn_sigmas/MatMul/ReadVariableOp�
MDN/mdn_sigmas/MatMulMatMulinputs,MDN/mdn_sigmas/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/MatMul�
%MDN/mdn_sigmas/BiasAdd/ReadVariableOpReadVariableOp.mdn_mdn_sigmas_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%MDN/mdn_sigmas/BiasAdd/ReadVariableOp�
MDN/mdn_sigmas/BiasAddBiasAddMDN/mdn_sigmas/MatMul:product:0-MDN/mdn_sigmas/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/BiasAdd�
MDN/mdn_sigmas/EluEluMDN/mdn_sigmas/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/Eluq
MDN/mdn_sigmas/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
MDN/mdn_sigmas/add/y�
MDN/mdn_sigmas/addAddV2 MDN/mdn_sigmas/Elu:activations:0MDN/mdn_sigmas/add/y:output:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/addu
MDN/mdn_sigmas/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
MDN/mdn_sigmas/add_1/y�
MDN/mdn_sigmas/add_1AddV2MDN/mdn_sigmas/add:z:0MDN/mdn_sigmas/add_1/y:output:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/add_1�
 MDN/mdn_pi/MatMul/ReadVariableOpReadVariableOp)mdn_mdn_pi_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype02"
 MDN/mdn_pi/MatMul/ReadVariableOp�
MDN/mdn_pi/MatMulMatMulinputs(MDN/mdn_pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MDN/mdn_pi/MatMul�
!MDN/mdn_pi/BiasAdd/ReadVariableOpReadVariableOp*mdn_mdn_pi_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02#
!MDN/mdn_pi/BiasAdd/ReadVariableOp�
MDN/mdn_pi/BiasAddBiasAddMDN/mdn_pi/MatMul:product:0)MDN/mdn_pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MDN/mdn_pi/BiasAdd|
MDN/mdn_outputs/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
MDN/mdn_outputs/concat/axis�
MDN/mdn_outputs/concatConcatV2MDN/mdn_mus/BiasAdd:output:0MDN/mdn_sigmas/add_1:z:0MDN/mdn_pi/BiasAdd:output:0$MDN/mdn_outputs/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
MDN/mdn_outputs/concat{
IdentityIdentityMDN/mdn_outputs/concat:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp#^MDN/mdn_mus/BiasAdd/ReadVariableOp"^MDN/mdn_mus/MatMul/ReadVariableOp"^MDN/mdn_pi/BiasAdd/ReadVariableOp!^MDN/mdn_pi/MatMul/ReadVariableOp&^MDN/mdn_sigmas/BiasAdd/ReadVariableOp%^MDN/mdn_sigmas/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2H
"MDN/mdn_mus/BiasAdd/ReadVariableOp"MDN/mdn_mus/BiasAdd/ReadVariableOp2F
!MDN/mdn_mus/MatMul/ReadVariableOp!MDN/mdn_mus/MatMul/ReadVariableOp2F
!MDN/mdn_pi/BiasAdd/ReadVariableOp!MDN/mdn_pi/BiasAdd/ReadVariableOp2D
 MDN/mdn_pi/MatMul/ReadVariableOp MDN/mdn_pi/MatMul/ReadVariableOp2N
%MDN/mdn_sigmas/BiasAdd/ReadVariableOp%MDN/mdn_sigmas/BiasAdd/ReadVariableOp2L
$MDN/mdn_sigmas/MatMul/ReadVariableOp$MDN/mdn_sigmas/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
F__inference_output_mdn_layer_call_and_return_conditional_losses_649628

inputs>
*mdn_mdn_mus_matmul_readvariableop_resource:
��:
+mdn_mdn_mus_biasadd_readvariableop_resource:	�A
-mdn_mdn_sigmas_matmul_readvariableop_resource:
��=
.mdn_mdn_sigmas_biasadd_readvariableop_resource:	�<
)mdn_mdn_pi_matmul_readvariableop_resource:	�28
*mdn_mdn_pi_biasadd_readvariableop_resource:2
identity��"MDN/mdn_mus/BiasAdd/ReadVariableOp�!MDN/mdn_mus/MatMul/ReadVariableOp�!MDN/mdn_pi/BiasAdd/ReadVariableOp� MDN/mdn_pi/MatMul/ReadVariableOp�%MDN/mdn_sigmas/BiasAdd/ReadVariableOp�$MDN/mdn_sigmas/MatMul/ReadVariableOp�
!MDN/mdn_mus/MatMul/ReadVariableOpReadVariableOp*mdn_mdn_mus_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!MDN/mdn_mus/MatMul/ReadVariableOp�
MDN/mdn_mus/MatMulMatMulinputs)MDN/mdn_mus/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/mdn_mus/MatMul�
"MDN/mdn_mus/BiasAdd/ReadVariableOpReadVariableOp+mdn_mdn_mus_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"MDN/mdn_mus/BiasAdd/ReadVariableOp�
MDN/mdn_mus/BiasAddBiasAddMDN/mdn_mus/MatMul:product:0*MDN/mdn_mus/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/mdn_mus/BiasAdd�
$MDN/mdn_sigmas/MatMul/ReadVariableOpReadVariableOp-mdn_mdn_sigmas_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02&
$MDN/mdn_sigmas/MatMul/ReadVariableOp�
MDN/mdn_sigmas/MatMulMatMulinputs,MDN/mdn_sigmas/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/MatMul�
%MDN/mdn_sigmas/BiasAdd/ReadVariableOpReadVariableOp.mdn_mdn_sigmas_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%MDN/mdn_sigmas/BiasAdd/ReadVariableOp�
MDN/mdn_sigmas/BiasAddBiasAddMDN/mdn_sigmas/MatMul:product:0-MDN/mdn_sigmas/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/BiasAdd�
MDN/mdn_sigmas/EluEluMDN/mdn_sigmas/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/Eluq
MDN/mdn_sigmas/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
MDN/mdn_sigmas/add/y�
MDN/mdn_sigmas/addAddV2 MDN/mdn_sigmas/Elu:activations:0MDN/mdn_sigmas/add/y:output:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/addu
MDN/mdn_sigmas/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
MDN/mdn_sigmas/add_1/y�
MDN/mdn_sigmas/add_1AddV2MDN/mdn_sigmas/add:z:0MDN/mdn_sigmas/add_1/y:output:0*
T0*(
_output_shapes
:����������2
MDN/mdn_sigmas/add_1�
 MDN/mdn_pi/MatMul/ReadVariableOpReadVariableOp)mdn_mdn_pi_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype02"
 MDN/mdn_pi/MatMul/ReadVariableOp�
MDN/mdn_pi/MatMulMatMulinputs(MDN/mdn_pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MDN/mdn_pi/MatMul�
!MDN/mdn_pi/BiasAdd/ReadVariableOpReadVariableOp*mdn_mdn_pi_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02#
!MDN/mdn_pi/BiasAdd/ReadVariableOp�
MDN/mdn_pi/BiasAddBiasAddMDN/mdn_pi/MatMul:product:0)MDN/mdn_pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MDN/mdn_pi/BiasAdd|
MDN/mdn_outputs/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
MDN/mdn_outputs/concat/axis�
MDN/mdn_outputs/concatConcatV2MDN/mdn_mus/BiasAdd:output:0MDN/mdn_sigmas/add_1:z:0MDN/mdn_pi/BiasAdd:output:0$MDN/mdn_outputs/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
MDN/mdn_outputs/concat{
IdentityIdentityMDN/mdn_outputs/concat:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp#^MDN/mdn_mus/BiasAdd/ReadVariableOp"^MDN/mdn_mus/MatMul/ReadVariableOp"^MDN/mdn_pi/BiasAdd/ReadVariableOp!^MDN/mdn_pi/MatMul/ReadVariableOp&^MDN/mdn_sigmas/BiasAdd/ReadVariableOp%^MDN/mdn_sigmas/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2H
"MDN/mdn_mus/BiasAdd/ReadVariableOp"MDN/mdn_mus/BiasAdd/ReadVariableOp2F
!MDN/mdn_mus/MatMul/ReadVariableOp!MDN/mdn_mus/MatMul/ReadVariableOp2F
!MDN/mdn_pi/BiasAdd/ReadVariableOp!MDN/mdn_pi/BiasAdd/ReadVariableOp2D
 MDN/mdn_pi/MatMul/ReadVariableOp MDN/mdn_pi/MatMul/ReadVariableOp2N
%MDN/mdn_sigmas/BiasAdd/ReadVariableOp%MDN/mdn_sigmas/BiasAdd/ReadVariableOp2L
$MDN/mdn_sigmas/MatMul/ReadVariableOp$MDN/mdn_sigmas/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_relu_2_layer_call_fn_650156

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_2_layer_call_and_return_conditional_losses_6495932
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
?__inference_MDN_layer_call_and_return_conditional_losses_649643

inputs 
relu_0_649560:	�
relu_0_649562:	�!
relu_1_649577:
��
relu_1_649579:	�!
relu_2_649594:
��
relu_2_649596:	�%
output_mdn_649629:
�� 
output_mdn_649631:	�%
output_mdn_649633:
�� 
output_mdn_649635:	�$
output_mdn_649637:	�2
output_mdn_649639:2
identity��"output_mdn/StatefulPartitionedCall�relu_0/StatefulPartitionedCall�relu_1/StatefulPartitionedCall�relu_2/StatefulPartitionedCall�
relu_0/StatefulPartitionedCallStatefulPartitionedCallinputsrelu_0_649560relu_0_649562*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_0_layer_call_and_return_conditional_losses_6495592 
relu_0/StatefulPartitionedCall�
relu_1/StatefulPartitionedCallStatefulPartitionedCall'relu_0/StatefulPartitionedCall:output:0relu_1_649577relu_1_649579*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_1_layer_call_and_return_conditional_losses_6495762 
relu_1/StatefulPartitionedCall�
relu_2/StatefulPartitionedCallStatefulPartitionedCall'relu_1/StatefulPartitionedCall:output:0relu_2_649594relu_2_649596*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_2_layer_call_and_return_conditional_losses_6495932 
relu_2/StatefulPartitionedCall�
"output_mdn/StatefulPartitionedCallStatefulPartitionedCall'relu_2/StatefulPartitionedCall:output:0output_mdn_649629output_mdn_649631output_mdn_649633output_mdn_649635output_mdn_649637output_mdn_649639*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *O
fJRH
F__inference_output_mdn_layer_call_and_return_conditional_losses_6496282$
"output_mdn/StatefulPartitionedCall�
IdentityIdentity+output_mdn/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp#^output_mdn/StatefulPartitionedCall^relu_0/StatefulPartitionedCall^relu_1/StatefulPartitionedCall^relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 2H
"output_mdn/StatefulPartitionedCall"output_mdn/StatefulPartitionedCall2@
relu_0/StatefulPartitionedCallrelu_0/StatefulPartitionedCall2@
relu_1/StatefulPartitionedCallrelu_1/StatefulPartitionedCall2@
relu_2/StatefulPartitionedCallrelu_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_relu_0_layer_call_fn_650116

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8� *K
fFRD
B__inference_relu_0_layer_call_and_return_conditional_losses_6495592
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
7
input.
serving_default_input:0���������?

output_mdn1
StatefulPartitionedCall:0����������tensorflow/serving/predict:�{
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_sequential
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
mdn_mus

mdn_sigmas

mdn_pi
 	variables
!trainable_variables
"regularization_losses
#	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemhmimjmkmlmm)mn*mo+mp,mq-mr.msvtvuvvvwvxvy)vz*v{+v|,v}-v~.v"
	optimizer
v
0
1
2
3
4
5
)6
*7
+8
,9
-10
.11"
trackable_list_wrapper
v
0
1
2
3
4
5
)6
*7
+8
,9
-10
.11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
/non_trainable_variables
0layer_regularization_losses

1layers
2metrics
3layer_metrics
trainable_variables
	variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 :	�2relu_0/kernel
:�2relu_0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
4non_trainable_variables
5layer_regularization_losses
6layer_metrics
7metrics
	variables
trainable_variables

8layers
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2relu_1/kernel
:�2relu_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9non_trainable_variables
:layer_regularization_losses
;layer_metrics
<metrics
	variables
trainable_variables

=layers
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2relu_2/kernel
:�2relu_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables
?layer_regularization_losses
@layer_metrics
Ametrics
	variables
trainable_variables

Blayers
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

)kernel
*bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

+kernel
,bias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

-kernel
.bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables
Player_regularization_losses
Qlayer_metrics
Rmetrics
 	variables
!trainable_variables

Slayers
"regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'
��2output_mdn/mus/kernel
": �2output_mdn/mus/bias
,:*
��2output_mdn/sigmas/kernel
%:#�2output_mdn/sigmas/bias
(:&	�22output_mdn/pis/kernel
!:22output_mdn/pis/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
T0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables
Vlayer_regularization_losses
Wlayer_metrics
Xmetrics
C	variables
Dtrainable_variables

Ylayers
Eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Znon_trainable_variables
[layer_regularization_losses
\layer_metrics
]metrics
G	variables
Htrainable_variables

^layers
Iregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables
`layer_regularization_losses
alayer_metrics
bmetrics
K	variables
Ltrainable_variables

clayers
Mregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
N
	dtotal
	ecount
f	variables
g	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
d0
e1"
trackable_list_wrapper
-
f	variables"
_generic_user_object
%:#	�2Adam/relu_0/kernel/m
:�2Adam/relu_0/bias/m
&:$
��2Adam/relu_1/kernel/m
:�2Adam/relu_1/bias/m
&:$
��2Adam/relu_2/kernel/m
:�2Adam/relu_2/bias/m
.:,
��2Adam/output_mdn/mus/kernel/m
':%�2Adam/output_mdn/mus/bias/m
1:/
��2Adam/output_mdn/sigmas/kernel/m
*:(�2Adam/output_mdn/sigmas/bias/m
-:+	�22Adam/output_mdn/pis/kernel/m
&:$22Adam/output_mdn/pis/bias/m
%:#	�2Adam/relu_0/kernel/v
:�2Adam/relu_0/bias/v
&:$
��2Adam/relu_1/kernel/v
:�2Adam/relu_1/bias/v
&:$
��2Adam/relu_2/kernel/v
:�2Adam/relu_2/bias/v
.:,
��2Adam/output_mdn/mus/kernel/v
':%�2Adam/output_mdn/mus/bias/v
1:/
��2Adam/output_mdn/sigmas/kernel/v
*:(�2Adam/output_mdn/sigmas/bias/v
-:+	�22Adam/output_mdn/pis/kernel/v
&:$22Adam/output_mdn/pis/bias/v
�B�
!__inference__wrapped_model_649541input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_MDN_layer_call_and_return_conditional_losses_649988
?__inference_MDN_layer_call_and_return_conditional_losses_650038
?__inference_MDN_layer_call_and_return_conditional_losses_649869
?__inference_MDN_layer_call_and_return_conditional_losses_649901�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_MDN_layer_call_fn_649670
$__inference_MDN_layer_call_fn_650067
$__inference_MDN_layer_call_fn_650096
$__inference_MDN_layer_call_fn_649837�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_relu_0_layer_call_and_return_conditional_losses_650107�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_relu_0_layer_call_fn_650116�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_relu_1_layer_call_and_return_conditional_losses_650127�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_relu_1_layer_call_fn_650136�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_relu_2_layer_call_and_return_conditional_losses_650147�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_relu_2_layer_call_fn_650156�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_output_mdn_layer_call_and_return_conditional_losses_650185�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_output_mdn_layer_call_fn_650202�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_649938input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
?__inference_MDN_layer_call_and_return_conditional_losses_649869n)*+,-.6�3
,�)
�
input���������
p 

 
� "&�#
�
0����������
� �
?__inference_MDN_layer_call_and_return_conditional_losses_649901n)*+,-.6�3
,�)
�
input���������
p

 
� "&�#
�
0����������
� �
?__inference_MDN_layer_call_and_return_conditional_losses_649988o)*+,-.7�4
-�*
 �
inputs���������
p 

 
� "&�#
�
0����������
� �
?__inference_MDN_layer_call_and_return_conditional_losses_650038o)*+,-.7�4
-�*
 �
inputs���������
p

 
� "&�#
�
0����������
� �
$__inference_MDN_layer_call_fn_649670a)*+,-.6�3
,�)
�
input���������
p 

 
� "������������
$__inference_MDN_layer_call_fn_649837a)*+,-.6�3
,�)
�
input���������
p

 
� "������������
$__inference_MDN_layer_call_fn_650067b)*+,-.7�4
-�*
 �
inputs���������
p 

 
� "������������
$__inference_MDN_layer_call_fn_650096b)*+,-.7�4
-�*
 �
inputs���������
p

 
� "������������
!__inference__wrapped_model_649541x)*+,-..�+
$�!
�
input���������
� "8�5
3

output_mdn%�"

output_mdn�����������
F__inference_output_mdn_layer_call_and_return_conditional_losses_650185b)*+,-.0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_output_mdn_layer_call_fn_650202U)*+,-.0�-
&�#
!�
inputs����������
� "������������
B__inference_relu_0_layer_call_and_return_conditional_losses_650107]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� {
'__inference_relu_0_layer_call_fn_650116P/�,
%�"
 �
inputs���������
� "������������
B__inference_relu_1_layer_call_and_return_conditional_losses_650127^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_relu_1_layer_call_fn_650136Q0�-
&�#
!�
inputs����������
� "������������
B__inference_relu_2_layer_call_and_return_conditional_losses_650147^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_relu_2_layer_call_fn_650156Q0�-
&�#
!�
inputs����������
� "������������
$__inference_signature_wrapper_649938�)*+,-.7�4
� 
-�*
(
input�
input���������"8�5
3

output_mdn%�"

output_mdn����������