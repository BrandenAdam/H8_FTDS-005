ߢ
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
Const
output"dtype"
valuetensor"
dtypetype
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
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
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ??
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
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
?
linear_block_2/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namelinear_block_2/dense_3/kernel
?
1linear_block_2/dense_3/kernel/Read/ReadVariableOpReadVariableOplinear_block_2/dense_3/kernel*
_output_shapes

:*
dtype0
?
linear_block_2/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelinear_block_2/dense_3/bias
?
/linear_block_2/dense_3/bias/Read/ReadVariableOpReadVariableOplinear_block_2/dense_3/bias*
_output_shapes
:*
dtype0
?
*linear_block_2/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*linear_block_2/batch_normalization_2/gamma
?
>linear_block_2/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp*linear_block_2/batch_normalization_2/gamma*
_output_shapes
:*
dtype0
?
)linear_block_2/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)linear_block_2/batch_normalization_2/beta
?
=linear_block_2/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp)linear_block_2/batch_normalization_2/beta*
_output_shapes
:*
dtype0
?
0linear_block_2/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20linear_block_2/batch_normalization_2/moving_mean
?
Dlinear_block_2/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp0linear_block_2/batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
?
4linear_block_2/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64linear_block_2/batch_normalization_2/moving_variance
?
Hlinear_block_2/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp4linear_block_2/batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
?
linear_block_3/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namelinear_block_3/dense_4/kernel
?
1linear_block_3/dense_4/kernel/Read/ReadVariableOpReadVariableOplinear_block_3/dense_4/kernel*
_output_shapes

:*
dtype0
?
linear_block_3/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelinear_block_3/dense_4/bias
?
/linear_block_3/dense_4/bias/Read/ReadVariableOpReadVariableOplinear_block_3/dense_4/bias*
_output_shapes
:*
dtype0
?
*linear_block_3/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*linear_block_3/batch_normalization_3/gamma
?
>linear_block_3/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp*linear_block_3/batch_normalization_3/gamma*
_output_shapes
:*
dtype0
?
)linear_block_3/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)linear_block_3/batch_normalization_3/beta
?
=linear_block_3/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp)linear_block_3/batch_normalization_3/beta*
_output_shapes
:*
dtype0
?
0linear_block_3/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20linear_block_3/batch_normalization_3/moving_mean
?
Dlinear_block_3/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp0linear_block_3/batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
?
4linear_block_3/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64linear_block_3/batch_normalization_3/moving_variance
?
Hlinear_block_3/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp4linear_block_3/batch_normalization_3/moving_variance*
_output_shapes
:*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_5/kernel/m
u
$dense_5/kernel/m/Read/ReadVariableOpReadVariableOpdense_5/kernel/m*
_output_shapes

:*
dtype0
t
dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias/m
m
"dense_5/bias/m/Read/ReadVariableOpReadVariableOpdense_5/bias/m*
_output_shapes
:*
dtype0
?
linear_block_2/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!linear_block_2/dense_3/kernel/m
?
3linear_block_2/dense_3/kernel/m/Read/ReadVariableOpReadVariableOplinear_block_2/dense_3/kernel/m*
_output_shapes

:*
dtype0
?
linear_block_2/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namelinear_block_2/dense_3/bias/m
?
1linear_block_2/dense_3/bias/m/Read/ReadVariableOpReadVariableOplinear_block_2/dense_3/bias/m*
_output_shapes
:*
dtype0
?
,linear_block_2/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,linear_block_2/batch_normalization_2/gamma/m
?
@linear_block_2/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp,linear_block_2/batch_normalization_2/gamma/m*
_output_shapes
:*
dtype0
?
+linear_block_2/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+linear_block_2/batch_normalization_2/beta/m
?
?linear_block_2/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp+linear_block_2/batch_normalization_2/beta/m*
_output_shapes
:*
dtype0
?
linear_block_3/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!linear_block_3/dense_4/kernel/m
?
3linear_block_3/dense_4/kernel/m/Read/ReadVariableOpReadVariableOplinear_block_3/dense_4/kernel/m*
_output_shapes

:*
dtype0
?
linear_block_3/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namelinear_block_3/dense_4/bias/m
?
1linear_block_3/dense_4/bias/m/Read/ReadVariableOpReadVariableOplinear_block_3/dense_4/bias/m*
_output_shapes
:*
dtype0
?
,linear_block_3/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,linear_block_3/batch_normalization_3/gamma/m
?
@linear_block_3/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp,linear_block_3/batch_normalization_3/gamma/m*
_output_shapes
:*
dtype0
?
+linear_block_3/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+linear_block_3/batch_normalization_3/beta/m
?
?linear_block_3/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp+linear_block_3/batch_normalization_3/beta/m*
_output_shapes
:*
dtype0
|
dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_5/kernel/v
u
$dense_5/kernel/v/Read/ReadVariableOpReadVariableOpdense_5/kernel/v*
_output_shapes

:*
dtype0
t
dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias/v
m
"dense_5/bias/v/Read/ReadVariableOpReadVariableOpdense_5/bias/v*
_output_shapes
:*
dtype0
?
linear_block_2/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!linear_block_2/dense_3/kernel/v
?
3linear_block_2/dense_3/kernel/v/Read/ReadVariableOpReadVariableOplinear_block_2/dense_3/kernel/v*
_output_shapes

:*
dtype0
?
linear_block_2/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namelinear_block_2/dense_3/bias/v
?
1linear_block_2/dense_3/bias/v/Read/ReadVariableOpReadVariableOplinear_block_2/dense_3/bias/v*
_output_shapes
:*
dtype0
?
,linear_block_2/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,linear_block_2/batch_normalization_2/gamma/v
?
@linear_block_2/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp,linear_block_2/batch_normalization_2/gamma/v*
_output_shapes
:*
dtype0
?
+linear_block_2/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+linear_block_2/batch_normalization_2/beta/v
?
?linear_block_2/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp+linear_block_2/batch_normalization_2/beta/v*
_output_shapes
:*
dtype0
?
linear_block_3/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!linear_block_3/dense_4/kernel/v
?
3linear_block_3/dense_4/kernel/v/Read/ReadVariableOpReadVariableOplinear_block_3/dense_4/kernel/v*
_output_shapes

:*
dtype0
?
linear_block_3/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namelinear_block_3/dense_4/bias/v
?
1linear_block_3/dense_4/bias/v/Read/ReadVariableOpReadVariableOplinear_block_3/dense_4/bias/v*
_output_shapes
:*
dtype0
?
,linear_block_3/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,linear_block_3/batch_normalization_3/gamma/v
?
@linear_block_3/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp,linear_block_3/batch_normalization_3/gamma/v*
_output_shapes
:*
dtype0
?
+linear_block_3/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+linear_block_3/batch_normalization_3/beta/v
?
?linear_block_3/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp+linear_block_3/batch_normalization_3/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
?M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?M
value?MB?M B?M
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
 
}

linear
bn
relu
dropout
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
}

linear
bn
relu
dropout
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?
&iter

'beta_1

(beta_2
	)decay
*learning_rate m?!m?+m?,m?-m?.m?1m?2m?3m?4m? v?!v?+v?,v?-v?.v?1v?2v?3v?4v?
f
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
 12
!13
F
+0
,1
-2
.3
14
25
36
47
 8
!9
 
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
	regularization_losses
 
h

+kernel
,bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?
@axis
	-gamma
.beta
/moving_mean
0moving_variance
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
R
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
*
+0
,1
-2
.3
/4
05

+0
,1
-2
.3
 
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
h

1kernel
2bias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?
[axis
	3gamma
4beta
5moving_mean
6moving_variance
\	variables
]trainable_variables
^regularization_losses
_	keras_api
R
`	variables
atrainable_variables
bregularization_losses
c	keras_api
R
d	variables
etrainable_variables
fregularization_losses
g	keras_api
*
10
21
32
43
54
65

10
21
32
43
 
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
"	variables
#trainable_variables
$regularization_losses
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
YW
VARIABLE_VALUElinear_block_2/dense_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElinear_block_2/dense_3/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*linear_block_2/batch_normalization_2/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)linear_block_2/batch_normalization_2/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0linear_block_2/batch_normalization_2/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE4linear_block_2/batch_normalization_2/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUElinear_block_3/dense_4/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElinear_block_3/dense_4/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*linear_block_3/batch_normalization_3/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)linear_block_3/batch_normalization_3/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0linear_block_3/batch_normalization_3/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4linear_block_3/batch_normalization_3/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE

/0
01
52
63
#
0
1
2
3
4

r0
s1
 
 

+0
,1

+0
,1
 
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
<	variables
=trainable_variables
>regularization_losses
 

-0
.1
/2
03

-0
.1
 
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
 
 
 
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses

/0
01

0
1
2
3
 
 
 
 
 
 
 
 

10
21

10
21
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
 

30
41
52
63

30
41
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses

50
61

0
1
2
3
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 

/0
01
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
 
 
 

50
61
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

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
xv
VARIABLE_VALUEdense_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUElinear_block_2/dense_3/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUElinear_block_2/dense_3/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,linear_block_2/batch_normalization_2/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+linear_block_2/batch_normalization_2/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUElinear_block_3/dense_4/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUElinear_block_3/dense_4/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,linear_block_3/batch_normalization_3/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+linear_block_3/batch_normalization_3/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUElinear_block_2/dense_3/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUElinear_block_2/dense_3/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,linear_block_2/batch_normalization_2/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+linear_block_2/batch_normalization_2/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUElinear_block_3/dense_4/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUElinear_block_3/dense_4/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,linear_block_3/batch_normalization_3/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+linear_block_3/batch_normalization_3/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2linear_block_2/dense_3/kernellinear_block_2/dense_3/bias4linear_block_2/batch_normalization_2/moving_variance*linear_block_2/batch_normalization_2/gamma0linear_block_2/batch_normalization_2/moving_mean)linear_block_2/batch_normalization_2/betalinear_block_3/dense_4/kernellinear_block_3/dense_4/bias4linear_block_3/batch_normalization_3/moving_variance*linear_block_3/batch_normalization_3/gamma0linear_block_3/batch_normalization_3/moving_mean)linear_block_3/batch_normalization_3/betadense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_8687739
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp1linear_block_2/dense_3/kernel/Read/ReadVariableOp/linear_block_2/dense_3/bias/Read/ReadVariableOp>linear_block_2/batch_normalization_2/gamma/Read/ReadVariableOp=linear_block_2/batch_normalization_2/beta/Read/ReadVariableOpDlinear_block_2/batch_normalization_2/moving_mean/Read/ReadVariableOpHlinear_block_2/batch_normalization_2/moving_variance/Read/ReadVariableOp1linear_block_3/dense_4/kernel/Read/ReadVariableOp/linear_block_3/dense_4/bias/Read/ReadVariableOp>linear_block_3/batch_normalization_3/gamma/Read/ReadVariableOp=linear_block_3/batch_normalization_3/beta/Read/ReadVariableOpDlinear_block_3/batch_normalization_3/moving_mean/Read/ReadVariableOpHlinear_block_3/batch_normalization_3/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp$dense_5/kernel/m/Read/ReadVariableOp"dense_5/bias/m/Read/ReadVariableOp3linear_block_2/dense_3/kernel/m/Read/ReadVariableOp1linear_block_2/dense_3/bias/m/Read/ReadVariableOp@linear_block_2/batch_normalization_2/gamma/m/Read/ReadVariableOp?linear_block_2/batch_normalization_2/beta/m/Read/ReadVariableOp3linear_block_3/dense_4/kernel/m/Read/ReadVariableOp1linear_block_3/dense_4/bias/m/Read/ReadVariableOp@linear_block_3/batch_normalization_3/gamma/m/Read/ReadVariableOp?linear_block_3/batch_normalization_3/beta/m/Read/ReadVariableOp$dense_5/kernel/v/Read/ReadVariableOp"dense_5/bias/v/Read/ReadVariableOp3linear_block_2/dense_3/kernel/v/Read/ReadVariableOp1linear_block_2/dense_3/bias/v/Read/ReadVariableOp@linear_block_2/batch_normalization_2/gamma/v/Read/ReadVariableOp?linear_block_2/batch_normalization_2/beta/v/Read/ReadVariableOp3linear_block_3/dense_4/kernel/v/Read/ReadVariableOp1linear_block_3/dense_4/bias/v/Read/ReadVariableOp@linear_block_3/batch_normalization_3/gamma/v/Read/ReadVariableOp?linear_block_3/batch_normalization_3/beta/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_8688611
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelinear_block_2/dense_3/kernellinear_block_2/dense_3/bias*linear_block_2/batch_normalization_2/gamma)linear_block_2/batch_normalization_2/beta0linear_block_2/batch_normalization_2/moving_mean4linear_block_2/batch_normalization_2/moving_variancelinear_block_3/dense_4/kernellinear_block_3/dense_4/bias*linear_block_3/batch_normalization_3/gamma)linear_block_3/batch_normalization_3/beta0linear_block_3/batch_normalization_3/moving_mean4linear_block_3/batch_normalization_3/moving_variancetotalcounttotal_1count_1dense_5/kernel/mdense_5/bias/mlinear_block_2/dense_3/kernel/mlinear_block_2/dense_3/bias/m,linear_block_2/batch_normalization_2/gamma/m+linear_block_2/batch_normalization_2/beta/mlinear_block_3/dense_4/kernel/mlinear_block_3/dense_4/bias/m,linear_block_3/batch_normalization_3/gamma/m+linear_block_3/batch_normalization_3/beta/mdense_5/kernel/vdense_5/bias/vlinear_block_2/dense_3/kernel/vlinear_block_2/dense_3/bias/v,linear_block_2/batch_normalization_2/gamma/v+linear_block_2/batch_normalization_2/beta/vlinear_block_3/dense_4/kernel/vlinear_block_3/dense_4/bias/v,linear_block_3/batch_normalization_3/gamma/v+linear_block_3/batch_normalization_3/beta/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_8688750??
?
?
)__inference_model_1_layer_call_fn_8687254
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_8687223o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8687024

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
D__inference_model_1_layer_call_and_return_conditional_losses_8687686
input_2(
linear_block_2_8687641:$
linear_block_2_8687643:$
linear_block_2_8687645:$
linear_block_2_8687647:$
linear_block_2_8687649:$
linear_block_2_8687651:(
linear_block_3_8687655:$
linear_block_3_8687657:$
linear_block_3_8687659:$
linear_block_3_8687661:$
linear_block_3_8687663:$
linear_block_3_8687665:!
dense_5_8687668:
dense_5_8687670:
identity??dense_5/StatefulPartitionedCall?&linear_block_2/StatefulPartitionedCall??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?&linear_block_3/StatefulPartitionedCall??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
&linear_block_2/StatefulPartitionedCallStatefulPartitionedCallinput_2linear_block_2_8687641linear_block_2_8687643linear_block_2_8687645linear_block_2_8687647linear_block_2_8687649linear_block_2_8687651*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8687428?
add_1/PartitionedCallPartitionedCall/linear_block_2/StatefulPartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_8687143?
&linear_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0linear_block_3_8687655linear_block_3_8687657linear_block_3_8687659linear_block_3_8687661linear_block_3_8687663linear_block_3_8687665*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8687339?
dense_5/StatefulPartitionedCallStatefulPartitionedCall/linear_block_3/StatefulPartitionedCall:output:0dense_5_8687668dense_5_8687670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_8687204?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOplinear_block_2_8687641*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplinear_block_3_8687655*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_5/StatefulPartitionedCall'^linear_block_2/StatefulPartitionedCall@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp'^linear_block_3/StatefulPartitionedCall@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2P
&linear_block_2/StatefulPartitionedCall&linear_block_2/StatefulPartitionedCall2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp2P
&linear_block_3/StatefulPartitionedCall&linear_block_3/StatefulPartitionedCall2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?%
?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8688448

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_linear_block_3_layer_call_fn_8688139

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8687179o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?Z
?
 __inference__traced_save_8688611
file_prefix-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop<
8savev2_linear_block_2_dense_3_kernel_read_readvariableop:
6savev2_linear_block_2_dense_3_bias_read_readvariableopI
Esavev2_linear_block_2_batch_normalization_2_gamma_read_readvariableopH
Dsavev2_linear_block_2_batch_normalization_2_beta_read_readvariableopO
Ksavev2_linear_block_2_batch_normalization_2_moving_mean_read_readvariableopS
Osavev2_linear_block_2_batch_normalization_2_moving_variance_read_readvariableop<
8savev2_linear_block_3_dense_4_kernel_read_readvariableop:
6savev2_linear_block_3_dense_4_bias_read_readvariableopI
Esavev2_linear_block_3_batch_normalization_3_gamma_read_readvariableopH
Dsavev2_linear_block_3_batch_normalization_3_beta_read_readvariableopO
Ksavev2_linear_block_3_batch_normalization_3_moving_mean_read_readvariableopS
Osavev2_linear_block_3_batch_normalization_3_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop/
+savev2_dense_5_kernel_m_read_readvariableop-
)savev2_dense_5_bias_m_read_readvariableop>
:savev2_linear_block_2_dense_3_kernel_m_read_readvariableop<
8savev2_linear_block_2_dense_3_bias_m_read_readvariableopK
Gsavev2_linear_block_2_batch_normalization_2_gamma_m_read_readvariableopJ
Fsavev2_linear_block_2_batch_normalization_2_beta_m_read_readvariableop>
:savev2_linear_block_3_dense_4_kernel_m_read_readvariableop<
8savev2_linear_block_3_dense_4_bias_m_read_readvariableopK
Gsavev2_linear_block_3_batch_normalization_3_gamma_m_read_readvariableopJ
Fsavev2_linear_block_3_batch_normalization_3_beta_m_read_readvariableop/
+savev2_dense_5_kernel_v_read_readvariableop-
)savev2_dense_5_bias_v_read_readvariableop>
:savev2_linear_block_2_dense_3_kernel_v_read_readvariableop<
8savev2_linear_block_2_dense_3_bias_v_read_readvariableopK
Gsavev2_linear_block_2_batch_normalization_2_gamma_v_read_readvariableopJ
Fsavev2_linear_block_2_batch_normalization_2_beta_v_read_readvariableop>
:savev2_linear_block_3_dense_4_kernel_v_read_readvariableop<
8savev2_linear_block_3_dense_4_bias_v_read_readvariableopK
Gsavev2_linear_block_3_batch_normalization_3_gamma_v_read_readvariableopJ
Fsavev2_linear_block_3_batch_normalization_3_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop8savev2_linear_block_2_dense_3_kernel_read_readvariableop6savev2_linear_block_2_dense_3_bias_read_readvariableopEsavev2_linear_block_2_batch_normalization_2_gamma_read_readvariableopDsavev2_linear_block_2_batch_normalization_2_beta_read_readvariableopKsavev2_linear_block_2_batch_normalization_2_moving_mean_read_readvariableopOsavev2_linear_block_2_batch_normalization_2_moving_variance_read_readvariableop8savev2_linear_block_3_dense_4_kernel_read_readvariableop6savev2_linear_block_3_dense_4_bias_read_readvariableopEsavev2_linear_block_3_batch_normalization_3_gamma_read_readvariableopDsavev2_linear_block_3_batch_normalization_3_beta_read_readvariableopKsavev2_linear_block_3_batch_normalization_3_moving_mean_read_readvariableopOsavev2_linear_block_3_batch_normalization_3_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop+savev2_dense_5_kernel_m_read_readvariableop)savev2_dense_5_bias_m_read_readvariableop:savev2_linear_block_2_dense_3_kernel_m_read_readvariableop8savev2_linear_block_2_dense_3_bias_m_read_readvariableopGsavev2_linear_block_2_batch_normalization_2_gamma_m_read_readvariableopFsavev2_linear_block_2_batch_normalization_2_beta_m_read_readvariableop:savev2_linear_block_3_dense_4_kernel_m_read_readvariableop8savev2_linear_block_3_dense_4_bias_m_read_readvariableopGsavev2_linear_block_3_batch_normalization_3_gamma_m_read_readvariableopFsavev2_linear_block_3_batch_normalization_3_beta_m_read_readvariableop+savev2_dense_5_kernel_v_read_readvariableop)savev2_dense_5_bias_v_read_readvariableop:savev2_linear_block_2_dense_3_kernel_v_read_readvariableop8savev2_linear_block_2_dense_3_bias_v_read_readvariableopGsavev2_linear_block_2_batch_normalization_2_gamma_v_read_readvariableopFsavev2_linear_block_2_batch_normalization_2_beta_v_read_readvariableop:savev2_linear_block_3_dense_4_kernel_v_read_readvariableop8savev2_linear_block_3_dense_4_bias_v_read_readvariableopGsavev2_linear_block_3_batch_normalization_3_gamma_v_read_readvariableopFsavev2_linear_block_3_batch_normalization_3_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : : : ::::::::::::: : : : ::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
::,

_output_shapes
: 
?%
?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8687071

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_8688750
file_prefix1
assignvariableop_dense_5_kernel:-
assignvariableop_1_dense_5_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: B
0assignvariableop_7_linear_block_2_dense_3_kernel:<
.assignvariableop_8_linear_block_2_dense_3_bias:K
=assignvariableop_9_linear_block_2_batch_normalization_2_gamma:K
=assignvariableop_10_linear_block_2_batch_normalization_2_beta:R
Dassignvariableop_11_linear_block_2_batch_normalization_2_moving_mean:V
Hassignvariableop_12_linear_block_2_batch_normalization_2_moving_variance:C
1assignvariableop_13_linear_block_3_dense_4_kernel:=
/assignvariableop_14_linear_block_3_dense_4_bias:L
>assignvariableop_15_linear_block_3_batch_normalization_3_gamma:K
=assignvariableop_16_linear_block_3_batch_normalization_3_beta:R
Dassignvariableop_17_linear_block_3_batch_normalization_3_moving_mean:V
Hassignvariableop_18_linear_block_3_batch_normalization_3_moving_variance:#
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: 6
$assignvariableop_23_dense_5_kernel_m:0
"assignvariableop_24_dense_5_bias_m:E
3assignvariableop_25_linear_block_2_dense_3_kernel_m:?
1assignvariableop_26_linear_block_2_dense_3_bias_m:N
@assignvariableop_27_linear_block_2_batch_normalization_2_gamma_m:M
?assignvariableop_28_linear_block_2_batch_normalization_2_beta_m:E
3assignvariableop_29_linear_block_3_dense_4_kernel_m:?
1assignvariableop_30_linear_block_3_dense_4_bias_m:N
@assignvariableop_31_linear_block_3_batch_normalization_3_gamma_m:M
?assignvariableop_32_linear_block_3_batch_normalization_3_beta_m:6
$assignvariableop_33_dense_5_kernel_v:0
"assignvariableop_34_dense_5_bias_v:E
3assignvariableop_35_linear_block_2_dense_3_kernel_v:?
1assignvariableop_36_linear_block_2_dense_3_bias_v:N
@assignvariableop_37_linear_block_2_batch_normalization_2_gamma_v:M
?assignvariableop_38_linear_block_2_batch_normalization_2_beta_v:E
3assignvariableop_39_linear_block_3_dense_4_kernel_v:?
1assignvariableop_40_linear_block_3_dense_4_bias_v:N
@assignvariableop_41_linear_block_3_batch_normalization_3_gamma_v:M
?assignvariableop_42_linear_block_3_batch_normalization_3_beta_v:
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp0assignvariableop_7_linear_block_2_dense_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_linear_block_2_dense_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp=assignvariableop_9_linear_block_2_batch_normalization_2_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp=assignvariableop_10_linear_block_2_batch_normalization_2_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpDassignvariableop_11_linear_block_2_batch_normalization_2_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpHassignvariableop_12_linear_block_2_batch_normalization_2_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp1assignvariableop_13_linear_block_3_dense_4_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_linear_block_3_dense_4_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp>assignvariableop_15_linear_block_3_batch_normalization_3_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp=assignvariableop_16_linear_block_3_batch_normalization_3_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpDassignvariableop_17_linear_block_3_batch_normalization_3_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpHassignvariableop_18_linear_block_3_batch_normalization_3_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp$assignvariableop_23_dense_5_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_5_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp3assignvariableop_25_linear_block_2_dense_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp1assignvariableop_26_linear_block_2_dense_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp@assignvariableop_27_linear_block_2_batch_normalization_2_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp?assignvariableop_28_linear_block_2_batch_normalization_2_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp3assignvariableop_29_linear_block_3_dense_4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp1assignvariableop_30_linear_block_3_dense_4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp@assignvariableop_31_linear_block_3_batch_normalization_3_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp?assignvariableop_32_linear_block_3_batch_normalization_3_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_5_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_5_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp3assignvariableop_35_linear_block_2_dense_3_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp1assignvariableop_36_linear_block_2_dense_3_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp@assignvariableop_37_linear_block_2_batch_normalization_2_gamma_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp?assignvariableop_38_linear_block_2_batch_normalization_2_beta_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp3assignvariableop_39_linear_block_3_dense_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp1assignvariableop_40_linear_block_3_dense_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp@assignvariableop_41_linear_block_3_batch_normalization_3_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp?assignvariableop_42_linear_block_3_batch_normalization_3_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
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
?G
?
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8688104

inputs8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:K
=batch_normalization_2_assignmovingavg_readvariableop_resource:M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:E
7batch_normalization_2_batchnorm_readvariableop_resource:
identity??%batch_normalization_2/AssignMovingAvg?4batch_normalization_2/AssignMovingAvg/ReadVariableOp?'batch_normalization_2/AssignMovingAvg_1?6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_2/batchnorm/ReadVariableOp?2batch_normalization_2/batchnorm/mul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_2/moments/meanMeandense_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:?
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:??????????
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:?
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_2/batchnorm/mul_1Muldense_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:?
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????v
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:??????????
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
B__inference_add_1_layer_call_and_return_conditional_losses_8687143

inputs
inputs_1
identityP
addAddV2inputsinputs_1*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?j
?
"__inference__wrapped_model_8686918
input_2O
=model_1_linear_block_2_dense_3_matmul_readvariableop_resource:L
>model_1_linear_block_2_dense_3_biasadd_readvariableop_resource:\
Nmodel_1_linear_block_2_batch_normalization_2_batchnorm_readvariableop_resource:`
Rmodel_1_linear_block_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:^
Pmodel_1_linear_block_2_batch_normalization_2_batchnorm_readvariableop_1_resource:^
Pmodel_1_linear_block_2_batch_normalization_2_batchnorm_readvariableop_2_resource:O
=model_1_linear_block_3_dense_4_matmul_readvariableop_resource:L
>model_1_linear_block_3_dense_4_biasadd_readvariableop_resource:\
Nmodel_1_linear_block_3_batch_normalization_3_batchnorm_readvariableop_resource:`
Rmodel_1_linear_block_3_batch_normalization_3_batchnorm_mul_readvariableop_resource:^
Pmodel_1_linear_block_3_batch_normalization_3_batchnorm_readvariableop_1_resource:^
Pmodel_1_linear_block_3_batch_normalization_3_batchnorm_readvariableop_2_resource:@
.model_1_dense_5_matmul_readvariableop_resource:=
/model_1_dense_5_biasadd_readvariableop_resource:
identity??&model_1/dense_5/BiasAdd/ReadVariableOp?%model_1/dense_5/MatMul/ReadVariableOp?Emodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp?Gmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1?Gmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2?Imodel_1/linear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp?5model_1/linear_block_2/dense_3/BiasAdd/ReadVariableOp?4model_1/linear_block_2/dense_3/MatMul/ReadVariableOp?Emodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp?Gmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1?Gmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2?Imodel_1/linear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp?5model_1/linear_block_3/dense_4/BiasAdd/ReadVariableOp?4model_1/linear_block_3/dense_4/MatMul/ReadVariableOp?
4model_1/linear_block_2/dense_3/MatMul/ReadVariableOpReadVariableOp=model_1_linear_block_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
%model_1/linear_block_2/dense_3/MatMulMatMulinput_2<model_1/linear_block_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
5model_1/linear_block_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp>model_1_linear_block_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&model_1/linear_block_2/dense_3/BiasAddBiasAdd/model_1/linear_block_2/dense_3/MatMul:product:0=model_1/linear_block_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Emodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpNmodel_1_linear_block_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
<model_1/linear_block_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
:model_1/linear_block_2/batch_normalization_2/batchnorm/addAddV2Mmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0Emodel_1/linear_block_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
<model_1/linear_block_2/batch_normalization_2/batchnorm/RsqrtRsqrt>model_1/linear_block_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:?
Imodel_1/linear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpRmodel_1_linear_block_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
:model_1/linear_block_2/batch_normalization_2/batchnorm/mulMul@model_1/linear_block_2/batch_normalization_2/batchnorm/Rsqrt:y:0Qmodel_1/linear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
<model_1/linear_block_2/batch_normalization_2/batchnorm/mul_1Mul/model_1/linear_block_2/dense_3/BiasAdd:output:0>model_1/linear_block_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
Gmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpPmodel_1_linear_block_2_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
<model_1/linear_block_2/batch_normalization_2/batchnorm/mul_2MulOmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0>model_1/linear_block_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:?
Gmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpPmodel_1_linear_block_2_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
:model_1/linear_block_2/batch_normalization_2/batchnorm/subSubOmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0@model_1/linear_block_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
<model_1/linear_block_2/batch_normalization_2/batchnorm/add_1AddV2@model_1/linear_block_2/batch_normalization_2/batchnorm/mul_1:z:0>model_1/linear_block_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
(model_1/linear_block_2/activation_2/ReluRelu@model_1/linear_block_2/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:??????????
)model_1/linear_block_2/dropout_2/IdentityIdentity6model_1/linear_block_2/activation_2/Relu:activations:0*
T0*'
_output_shapes
:??????????
model_1/add_1/addAddV22model_1/linear_block_2/dropout_2/Identity:output:0input_2*
T0*'
_output_shapes
:??????????
4model_1/linear_block_3/dense_4/MatMul/ReadVariableOpReadVariableOp=model_1_linear_block_3_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
%model_1/linear_block_3/dense_4/MatMulMatMulmodel_1/add_1/add:z:0<model_1/linear_block_3/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
5model_1/linear_block_3/dense_4/BiasAdd/ReadVariableOpReadVariableOp>model_1_linear_block_3_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&model_1/linear_block_3/dense_4/BiasAddBiasAdd/model_1/linear_block_3/dense_4/MatMul:product:0=model_1/linear_block_3/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Emodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpNmodel_1_linear_block_3_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
<model_1/linear_block_3/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
:model_1/linear_block_3/batch_normalization_3/batchnorm/addAddV2Mmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp:value:0Emodel_1/linear_block_3/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
<model_1/linear_block_3/batch_normalization_3/batchnorm/RsqrtRsqrt>model_1/linear_block_3/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:?
Imodel_1/linear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpRmodel_1_linear_block_3_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
:model_1/linear_block_3/batch_normalization_3/batchnorm/mulMul@model_1/linear_block_3/batch_normalization_3/batchnorm/Rsqrt:y:0Qmodel_1/linear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
<model_1/linear_block_3/batch_normalization_3/batchnorm/mul_1Mul/model_1/linear_block_3/dense_4/BiasAdd:output:0>model_1/linear_block_3/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
Gmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpPmodel_1_linear_block_3_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
<model_1/linear_block_3/batch_normalization_3/batchnorm/mul_2MulOmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0>model_1/linear_block_3/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:?
Gmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpPmodel_1_linear_block_3_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
:model_1/linear_block_3/batch_normalization_3/batchnorm/subSubOmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0@model_1/linear_block_3/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
<model_1/linear_block_3/batch_normalization_3/batchnorm/add_1AddV2@model_1/linear_block_3/batch_normalization_3/batchnorm/mul_1:z:0>model_1/linear_block_3/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
(model_1/linear_block_3/activation_3/ReluRelu@model_1/linear_block_3/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:??????????
)model_1/linear_block_3/dropout_3/IdentityIdentity6model_1/linear_block_3/activation_3/Relu:activations:0*
T0*'
_output_shapes
:??????????
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_1/dense_5/MatMulMatMul2model_1/linear_block_3/dropout_3/Identity:output:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
model_1/dense_5/SigmoidSigmoid model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????j
IdentityIdentitymodel_1/dense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOpF^model_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOpH^model_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1H^model_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2J^model_1/linear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp6^model_1/linear_block_2/dense_3/BiasAdd/ReadVariableOp5^model_1/linear_block_2/dense_3/MatMul/ReadVariableOpF^model_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOpH^model_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1H^model_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2J^model_1/linear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp6^model_1/linear_block_3/dense_4/BiasAdd/ReadVariableOp5^model_1/linear_block_3/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2?
Emodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOpEmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp2?
Gmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1Gmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_12?
Gmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2Gmodel_1/linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_22?
Imodel_1/linear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOpImodel_1/linear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp2n
5model_1/linear_block_2/dense_3/BiasAdd/ReadVariableOp5model_1/linear_block_2/dense_3/BiasAdd/ReadVariableOp2l
4model_1/linear_block_2/dense_3/MatMul/ReadVariableOp4model_1/linear_block_2/dense_3/MatMul/ReadVariableOp2?
Emodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOpEmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp2?
Gmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1Gmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_12?
Gmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2Gmodel_1/linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_22?
Imodel_1/linear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOpImodel_1/linear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp2n
5model_1/linear_block_3/dense_4/BiasAdd/ReadVariableOp5model_1/linear_block_3/dense_4/BiasAdd/ReadVariableOp2l
4model_1/linear_block_3/dense_4/MatMul/ReadVariableOp4model_1/linear_block_3/dense_4/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?P
?
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8687339

inputs8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:K
=batch_normalization_3_assignmovingavg_readvariableop_resource:M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:E
7batch_normalization_3_batchnorm_readvariableop_resource:
identity??%batch_normalization_3/AssignMovingAvg?4batch_normalization_3/AssignMovingAvg/ReadVariableOp?'batch_normalization_3/AssignMovingAvg_1?6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_3/batchnorm/ReadVariableOp?2batch_normalization_3/batchnorm/mul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_3/moments/meanMeandense_4/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:?
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_4/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:??????????
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:?
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_3/batchnorm/mul_1Muldense_4/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:?
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????v
activation_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_3/dropout/MulMulactivation_3/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????f
dropout_3/dropout/ShapeShapeactivation_3/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed{e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentitydropout_3/dropout/Mul_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_model_1_layer_call_fn_8687590
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_8687526o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
??
?
D__inference_model_1_layer_call_and_return_conditional_losses_8687983

inputsG
5linear_block_2_dense_3_matmul_readvariableop_resource:D
6linear_block_2_dense_3_biasadd_readvariableop_resource:Z
Llinear_block_2_batch_normalization_2_assignmovingavg_readvariableop_resource:\
Nlinear_block_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource:X
Jlinear_block_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:T
Flinear_block_2_batch_normalization_2_batchnorm_readvariableop_resource:G
5linear_block_3_dense_4_matmul_readvariableop_resource:D
6linear_block_3_dense_4_biasadd_readvariableop_resource:Z
Llinear_block_3_batch_normalization_3_assignmovingavg_readvariableop_resource:\
Nlinear_block_3_batch_normalization_3_assignmovingavg_1_readvariableop_resource:X
Jlinear_block_3_batch_normalization_3_batchnorm_mul_readvariableop_resource:T
Flinear_block_3_batch_normalization_3_batchnorm_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?4linear_block_2/batch_normalization_2/AssignMovingAvg?Clinear_block_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp?6linear_block_2/batch_normalization_2/AssignMovingAvg_1?Elinear_block_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?=linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp?Alinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp?-linear_block_2/dense_3/BiasAdd/ReadVariableOp?,linear_block_2/dense_3/MatMul/ReadVariableOp??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?4linear_block_3/batch_normalization_3/AssignMovingAvg?Clinear_block_3/batch_normalization_3/AssignMovingAvg/ReadVariableOp?6linear_block_3/batch_normalization_3/AssignMovingAvg_1?Elinear_block_3/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp?=linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp?Alinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp?-linear_block_3/dense_4/BiasAdd/ReadVariableOp?,linear_block_3/dense_4/MatMul/ReadVariableOp??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
,linear_block_2/dense_3/MatMul/ReadVariableOpReadVariableOp5linear_block_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
linear_block_2/dense_3/MatMulMatMulinputs4linear_block_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-linear_block_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp6linear_block_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
linear_block_2/dense_3/BiasAddBiasAdd'linear_block_2/dense_3/MatMul:product:05linear_block_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Clinear_block_2/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
1linear_block_2/batch_normalization_2/moments/meanMean'linear_block_2/dense_3/BiasAdd:output:0Llinear_block_2/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
9linear_block_2/batch_normalization_2/moments/StopGradientStopGradient:linear_block_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:?
>linear_block_2/batch_normalization_2/moments/SquaredDifferenceSquaredDifference'linear_block_2/dense_3/BiasAdd:output:0Blinear_block_2/batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:??????????
Glinear_block_2/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
5linear_block_2/batch_normalization_2/moments/varianceMeanBlinear_block_2/batch_normalization_2/moments/SquaredDifference:z:0Plinear_block_2/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
4linear_block_2/batch_normalization_2/moments/SqueezeSqueeze:linear_block_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
6linear_block_2/batch_normalization_2/moments/Squeeze_1Squeeze>linear_block_2/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 
:linear_block_2/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Clinear_block_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpLlinear_block_2_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
8linear_block_2/batch_normalization_2/AssignMovingAvg/subSubKlinear_block_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0=linear_block_2/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:?
8linear_block_2/batch_normalization_2/AssignMovingAvg/mulMul<linear_block_2/batch_normalization_2/AssignMovingAvg/sub:z:0Clinear_block_2/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
4linear_block_2/batch_normalization_2/AssignMovingAvgAssignSubVariableOpLlinear_block_2_batch_normalization_2_assignmovingavg_readvariableop_resource<linear_block_2/batch_normalization_2/AssignMovingAvg/mul:z:0D^linear_block_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0?
<linear_block_2/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Elinear_block_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpNlinear_block_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
:linear_block_2/batch_normalization_2/AssignMovingAvg_1/subSubMlinear_block_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0?linear_block_2/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
:linear_block_2/batch_normalization_2/AssignMovingAvg_1/mulMul>linear_block_2/batch_normalization_2/AssignMovingAvg_1/sub:z:0Elinear_block_2/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
6linear_block_2/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpNlinear_block_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource>linear_block_2/batch_normalization_2/AssignMovingAvg_1/mul:z:0F^linear_block_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0y
4linear_block_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
2linear_block_2/batch_normalization_2/batchnorm/addAddV2?linear_block_2/batch_normalization_2/moments/Squeeze_1:output:0=linear_block_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
4linear_block_2/batch_normalization_2/batchnorm/RsqrtRsqrt6linear_block_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:?
Alinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpJlinear_block_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
2linear_block_2/batch_normalization_2/batchnorm/mulMul8linear_block_2/batch_normalization_2/batchnorm/Rsqrt:y:0Ilinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
4linear_block_2/batch_normalization_2/batchnorm/mul_1Mul'linear_block_2/dense_3/BiasAdd:output:06linear_block_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
4linear_block_2/batch_normalization_2/batchnorm/mul_2Mul=linear_block_2/batch_normalization_2/moments/Squeeze:output:06linear_block_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:?
=linear_block_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpFlinear_block_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
2linear_block_2/batch_normalization_2/batchnorm/subSubElinear_block_2/batch_normalization_2/batchnorm/ReadVariableOp:value:08linear_block_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
4linear_block_2/batch_normalization_2/batchnorm/add_1AddV28linear_block_2/batch_normalization_2/batchnorm/mul_1:z:06linear_block_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
 linear_block_2/activation_2/ReluRelu8linear_block_2/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????|
	add_1/addAddV2.linear_block_2/activation_2/Relu:activations:0inputs*
T0*'
_output_shapes
:??????????
,linear_block_3/dense_4/MatMul/ReadVariableOpReadVariableOp5linear_block_3_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
linear_block_3/dense_4/MatMulMatMuladd_1/add:z:04linear_block_3/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-linear_block_3/dense_4/BiasAdd/ReadVariableOpReadVariableOp6linear_block_3_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
linear_block_3/dense_4/BiasAddBiasAdd'linear_block_3/dense_4/MatMul:product:05linear_block_3/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Clinear_block_3/batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
1linear_block_3/batch_normalization_3/moments/meanMean'linear_block_3/dense_4/BiasAdd:output:0Llinear_block_3/batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
9linear_block_3/batch_normalization_3/moments/StopGradientStopGradient:linear_block_3/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:?
>linear_block_3/batch_normalization_3/moments/SquaredDifferenceSquaredDifference'linear_block_3/dense_4/BiasAdd:output:0Blinear_block_3/batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:??????????
Glinear_block_3/batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
5linear_block_3/batch_normalization_3/moments/varianceMeanBlinear_block_3/batch_normalization_3/moments/SquaredDifference:z:0Plinear_block_3/batch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
4linear_block_3/batch_normalization_3/moments/SqueezeSqueeze:linear_block_3/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
6linear_block_3/batch_normalization_3/moments/Squeeze_1Squeeze>linear_block_3/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 
:linear_block_3/batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Clinear_block_3/batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOpLlinear_block_3_batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
8linear_block_3/batch_normalization_3/AssignMovingAvg/subSubKlinear_block_3/batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0=linear_block_3/batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:?
8linear_block_3/batch_normalization_3/AssignMovingAvg/mulMul<linear_block_3/batch_normalization_3/AssignMovingAvg/sub:z:0Clinear_block_3/batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
4linear_block_3/batch_normalization_3/AssignMovingAvgAssignSubVariableOpLlinear_block_3_batch_normalization_3_assignmovingavg_readvariableop_resource<linear_block_3/batch_normalization_3/AssignMovingAvg/mul:z:0D^linear_block_3/batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0?
<linear_block_3/batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Elinear_block_3/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOpNlinear_block_3_batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
:linear_block_3/batch_normalization_3/AssignMovingAvg_1/subSubMlinear_block_3/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0?linear_block_3/batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
:linear_block_3/batch_normalization_3/AssignMovingAvg_1/mulMul>linear_block_3/batch_normalization_3/AssignMovingAvg_1/sub:z:0Elinear_block_3/batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
6linear_block_3/batch_normalization_3/AssignMovingAvg_1AssignSubVariableOpNlinear_block_3_batch_normalization_3_assignmovingavg_1_readvariableop_resource>linear_block_3/batch_normalization_3/AssignMovingAvg_1/mul:z:0F^linear_block_3/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0y
4linear_block_3/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
2linear_block_3/batch_normalization_3/batchnorm/addAddV2?linear_block_3/batch_normalization_3/moments/Squeeze_1:output:0=linear_block_3/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
4linear_block_3/batch_normalization_3/batchnorm/RsqrtRsqrt6linear_block_3/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:?
Alinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpJlinear_block_3_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
2linear_block_3/batch_normalization_3/batchnorm/mulMul8linear_block_3/batch_normalization_3/batchnorm/Rsqrt:y:0Ilinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
4linear_block_3/batch_normalization_3/batchnorm/mul_1Mul'linear_block_3/dense_4/BiasAdd:output:06linear_block_3/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
4linear_block_3/batch_normalization_3/batchnorm/mul_2Mul=linear_block_3/batch_normalization_3/moments/Squeeze:output:06linear_block_3/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:?
=linear_block_3/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpFlinear_block_3_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
2linear_block_3/batch_normalization_3/batchnorm/subSubElinear_block_3/batch_normalization_3/batchnorm/ReadVariableOp:value:08linear_block_3/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
4linear_block_3/batch_normalization_3/batchnorm/add_1AddV28linear_block_3/batch_normalization_3/batchnorm/mul_1:z:06linear_block_3/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
 linear_block_3/activation_3/ReluRelu8linear_block_3/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????k
&linear_block_3/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$linear_block_3/dropout_3/dropout/MulMul.linear_block_3/activation_3/Relu:activations:0/linear_block_3/dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:??????????
&linear_block_3/dropout_3/dropout/ShapeShape.linear_block_3/activation_3/Relu:activations:0*
T0*
_output_shapes
:?
=linear_block_3/dropout_3/dropout/random_uniform/RandomUniformRandomUniform/linear_block_3/dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed{*
seed2t
/linear_block_3/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-linear_block_3/dropout_3/dropout/GreaterEqualGreaterEqualFlinear_block_3/dropout_3/dropout/random_uniform/RandomUniform:output:08linear_block_3/dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
%linear_block_3/dropout_3/dropout/CastCast1linear_block_3/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
&linear_block_3/dropout_3/dropout/Mul_1Mul(linear_block_3/dropout_3/dropout/Mul:z:0)linear_block_3/dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_5/MatMulMatMul*linear_block_3/dropout_3/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5linear_block_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5linear_block_3_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp5^linear_block_2/batch_normalization_2/AssignMovingAvgD^linear_block_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp7^linear_block_2/batch_normalization_2/AssignMovingAvg_1F^linear_block_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp>^linear_block_2/batch_normalization_2/batchnorm/ReadVariableOpB^linear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp.^linear_block_2/dense_3/BiasAdd/ReadVariableOp-^linear_block_2/dense_3/MatMul/ReadVariableOp@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp5^linear_block_3/batch_normalization_3/AssignMovingAvgD^linear_block_3/batch_normalization_3/AssignMovingAvg/ReadVariableOp7^linear_block_3/batch_normalization_3/AssignMovingAvg_1F^linear_block_3/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp>^linear_block_3/batch_normalization_3/batchnorm/ReadVariableOpB^linear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp.^linear_block_3/dense_4/BiasAdd/ReadVariableOp-^linear_block_3/dense_4/MatMul/ReadVariableOp@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2l
4linear_block_2/batch_normalization_2/AssignMovingAvg4linear_block_2/batch_normalization_2/AssignMovingAvg2?
Clinear_block_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpClinear_block_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp2p
6linear_block_2/batch_normalization_2/AssignMovingAvg_16linear_block_2/batch_normalization_2/AssignMovingAvg_12?
Elinear_block_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpElinear_block_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2~
=linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp=linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp2?
Alinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOpAlinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp2^
-linear_block_2/dense_3/BiasAdd/ReadVariableOp-linear_block_2/dense_3/BiasAdd/ReadVariableOp2\
,linear_block_2/dense_3/MatMul/ReadVariableOp,linear_block_2/dense_3/MatMul/ReadVariableOp2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp2l
4linear_block_3/batch_normalization_3/AssignMovingAvg4linear_block_3/batch_normalization_3/AssignMovingAvg2?
Clinear_block_3/batch_normalization_3/AssignMovingAvg/ReadVariableOpClinear_block_3/batch_normalization_3/AssignMovingAvg/ReadVariableOp2p
6linear_block_3/batch_normalization_3/AssignMovingAvg_16linear_block_3/batch_normalization_3/AssignMovingAvg_12?
Elinear_block_3/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpElinear_block_3/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2~
=linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp=linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp2?
Alinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOpAlinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp2^
-linear_block_3/dense_4/BiasAdd/ReadVariableOp-linear_block_3/dense_4/BiasAdd/ReadVariableOp2\
,linear_block_3/dense_4/MatMul/ReadVariableOp,linear_block_3/dense_4/MatMul/ReadVariableOp2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_3_layer_call_fn_8688381

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8687024o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
D__inference_model_1_layer_call_and_return_conditional_losses_8687526

inputs(
linear_block_2_8687481:$
linear_block_2_8687483:$
linear_block_2_8687485:$
linear_block_2_8687487:$
linear_block_2_8687489:$
linear_block_2_8687491:(
linear_block_3_8687495:$
linear_block_3_8687497:$
linear_block_3_8687499:$
linear_block_3_8687501:$
linear_block_3_8687503:$
linear_block_3_8687505:!
dense_5_8687508:
dense_5_8687510:
identity??dense_5/StatefulPartitionedCall?&linear_block_2/StatefulPartitionedCall??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?&linear_block_3/StatefulPartitionedCall??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
&linear_block_2/StatefulPartitionedCallStatefulPartitionedCallinputslinear_block_2_8687481linear_block_2_8687483linear_block_2_8687485linear_block_2_8687487linear_block_2_8687489linear_block_2_8687491*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8687428?
add_1/PartitionedCallPartitionedCall/linear_block_2/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_8687143?
&linear_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0linear_block_3_8687495linear_block_3_8687497linear_block_3_8687499linear_block_3_8687501linear_block_3_8687503linear_block_3_8687505*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8687339?
dense_5/StatefulPartitionedCallStatefulPartitionedCall/linear_block_3/StatefulPartitionedCall:output:0dense_5_8687508dense_5_8687510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_8687204?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOplinear_block_2_8687481*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplinear_block_3_8687495*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_5/StatefulPartitionedCall'^linear_block_2/StatefulPartitionedCall@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp'^linear_block_3/StatefulPartitionedCall@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2P
&linear_block_2/StatefulPartitionedCall&linear_block_2/StatefulPartitionedCall2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp2P
&linear_block_3/StatefulPartitionedCall&linear_block_3/StatefulPartitionedCall2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8688351

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
D__inference_model_1_layer_call_and_return_conditional_losses_8687638
input_2(
linear_block_2_8687593:$
linear_block_2_8687595:$
linear_block_2_8687597:$
linear_block_2_8687599:$
linear_block_2_8687601:$
linear_block_2_8687603:(
linear_block_3_8687607:$
linear_block_3_8687609:$
linear_block_3_8687611:$
linear_block_3_8687613:$
linear_block_3_8687615:$
linear_block_3_8687617:!
dense_5_8687620:
dense_5_8687622:
identity??dense_5/StatefulPartitionedCall?&linear_block_2/StatefulPartitionedCall??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?&linear_block_3/StatefulPartitionedCall??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
&linear_block_2/StatefulPartitionedCallStatefulPartitionedCallinput_2linear_block_2_8687593linear_block_2_8687595linear_block_2_8687597linear_block_2_8687599linear_block_2_8687601linear_block_2_8687603*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8687123?
add_1/PartitionedCallPartitionedCall/linear_block_2/StatefulPartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_8687143?
&linear_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0linear_block_3_8687607linear_block_3_8687609linear_block_3_8687611linear_block_3_8687613linear_block_3_8687615linear_block_3_8687617*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8687179?
dense_5/StatefulPartitionedCallStatefulPartitionedCall/linear_block_3/StatefulPartitionedCall:output:0dense_5_8687620dense_5_8687622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_8687204?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOplinear_block_2_8687593*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplinear_block_3_8687607*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_5/StatefulPartitionedCall'^linear_block_2/StatefulPartitionedCall@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp'^linear_block_3/StatefulPartitionedCall@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2P
&linear_block_2/StatefulPartitionedCall&linear_block_2/StatefulPartitionedCall2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp2P
&linear_block_3/StatefulPartitionedCall&linear_block_3/StatefulPartitionedCall2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
D__inference_dense_5_layer_call_and_return_conditional_losses_8688265

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8686942

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_3_layer_call_fn_8688394

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8687071o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8688414

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_model_1_layer_call_fn_8687772

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_8687223o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
D__inference_model_1_layer_call_and_return_conditional_losses_8687223

inputs(
linear_block_2_8687124:$
linear_block_2_8687126:$
linear_block_2_8687128:$
linear_block_2_8687130:$
linear_block_2_8687132:$
linear_block_2_8687134:(
linear_block_3_8687180:$
linear_block_3_8687182:$
linear_block_3_8687184:$
linear_block_3_8687186:$
linear_block_3_8687188:$
linear_block_3_8687190:!
dense_5_8687205:
dense_5_8687207:
identity??dense_5/StatefulPartitionedCall?&linear_block_2/StatefulPartitionedCall??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?&linear_block_3/StatefulPartitionedCall??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
&linear_block_2/StatefulPartitionedCallStatefulPartitionedCallinputslinear_block_2_8687124linear_block_2_8687126linear_block_2_8687128linear_block_2_8687130linear_block_2_8687132linear_block_2_8687134*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8687123?
add_1/PartitionedCallPartitionedCall/linear_block_2/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_8687143?
&linear_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0linear_block_3_8687180linear_block_3_8687182linear_block_3_8687184linear_block_3_8687186linear_block_3_8687188linear_block_3_8687190*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8687179?
dense_5/StatefulPartitionedCallStatefulPartitionedCall/linear_block_3/StatefulPartitionedCall:output:0dense_5_8687205dense_5_8687207*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_8687204?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOplinear_block_2_8687124*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplinear_block_3_8687180*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_5/StatefulPartitionedCall'^linear_block_2/StatefulPartitionedCall@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp'^linear_block_3/StatefulPartitionedCall@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2P
&linear_block_2/StatefulPartitionedCall&linear_block_2/StatefulPartitionedCall2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp2P
&linear_block_3/StatefulPartitionedCall&linear_block_3/StatefulPartitionedCall2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?G
?
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8687428

inputs8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:K
=batch_normalization_2_assignmovingavg_readvariableop_resource:M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:E
7batch_normalization_2_batchnorm_readvariableop_resource:
identity??%batch_normalization_2/AssignMovingAvg?4batch_normalization_2/AssignMovingAvg/ReadVariableOp?'batch_normalization_2/AssignMovingAvg_1?6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_2/batchnorm/ReadVariableOp?2batch_normalization_2/batchnorm/mul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_2/moments/meanMeandense_3/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:?
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_3/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:??????????
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:?
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_2/batchnorm/mul_1Muldense_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:?
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????v
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:??????????
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityactivation_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_linear_block_2_layer_call_fn_8688006

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8687123o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_8688362Z
Hlinear_block_2_dense_3_kernel_regularizer_square_readvariableop_resource:
identity???linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpHlinear_block_2_dense_3_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity1linear_block_2/dense_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp
?.
?
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8687123

inputs8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:E
7batch_normalization_2_batchnorm_readvariableop_resource:I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:G
9batch_normalization_2_batchnorm_readvariableop_1_resource:G
9batch_normalization_2_batchnorm_readvariableop_2_resource:
identity??.batch_normalization_2/batchnorm/ReadVariableOp?0batch_normalization_2/batchnorm/ReadVariableOp_1?0batch_normalization_2/batchnorm/ReadVariableOp_2?2batch_normalization_2/batchnorm/mul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_2/batchnorm/mul_1Muldense_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:?
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????v
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????q
dropout_2/IdentityIdentityactivation_2/Relu:activations:0*
T0*'
_output_shapes
:??????????
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentitydropout_2/Identity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8687179

inputs8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:E
7batch_normalization_3_batchnorm_readvariableop_resource:I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:G
9batch_normalization_3_batchnorm_readvariableop_1_resource:G
9batch_normalization_3_batchnorm_readvariableop_2_resource:
identity??.batch_normalization_3/batchnorm/ReadVariableOp?0batch_normalization_3/batchnorm/ReadVariableOp_1?0batch_normalization_3/batchnorm/ReadVariableOp_2?2batch_normalization_3/batchnorm/mul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_3/batchnorm/mul_1Muldense_4/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:?
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????v
activation_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????q
dropout_3/IdentityIdentityactivation_3/Relu:activations:0*
T0*'
_output_shapes
:??????????
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentitydropout_3/Identity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_linear_block_2_layer_call_fn_8688023

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8687428o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
S
'__inference_add_1_layer_call_fn_8688110
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_8687143`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
)__inference_dense_5_layer_call_fn_8688254

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_8687204o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8686989

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8688317

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_model_1_layer_call_fn_8687805

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_8687526o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_8688459Z
Hlinear_block_3_dense_4_kernel_regularizer_square_readvariableop_resource:
identity???linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpHlinear_block_3_dense_4_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity1linear_block_3/dense_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp
?

?
D__inference_dense_5_layer_call_and_return_conditional_losses_8687204

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8688190

inputs8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:E
7batch_normalization_3_batchnorm_readvariableop_resource:I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:G
9batch_normalization_3_batchnorm_readvariableop_1_resource:G
9batch_normalization_3_batchnorm_readvariableop_2_resource:
identity??.batch_normalization_3/batchnorm/ReadVariableOp?0batch_normalization_3/batchnorm/ReadVariableOp_1?0batch_normalization_3/batchnorm/ReadVariableOp_2?2batch_normalization_3/batchnorm/mul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_3/batchnorm/mul_1Muldense_4/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:?
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????v
activation_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????q
dropout_3/IdentityIdentityactivation_3/Relu:activations:0*
T0*'
_output_shapes
:??????????
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentitydropout_3/Identity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
B__inference_add_1_layer_call_and_return_conditional_losses_8688116
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?t
?
D__inference_model_1_layer_call_and_return_conditional_losses_8687877

inputsG
5linear_block_2_dense_3_matmul_readvariableop_resource:D
6linear_block_2_dense_3_biasadd_readvariableop_resource:T
Flinear_block_2_batch_normalization_2_batchnorm_readvariableop_resource:X
Jlinear_block_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:V
Hlinear_block_2_batch_normalization_2_batchnorm_readvariableop_1_resource:V
Hlinear_block_2_batch_normalization_2_batchnorm_readvariableop_2_resource:G
5linear_block_3_dense_4_matmul_readvariableop_resource:D
6linear_block_3_dense_4_biasadd_readvariableop_resource:T
Flinear_block_3_batch_normalization_3_batchnorm_readvariableop_resource:X
Jlinear_block_3_batch_normalization_3_batchnorm_mul_readvariableop_resource:V
Hlinear_block_3_batch_normalization_3_batchnorm_readvariableop_1_resource:V
Hlinear_block_3_batch_normalization_3_batchnorm_readvariableop_2_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?=linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp??linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1??linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2?Alinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp?-linear_block_2/dense_3/BiasAdd/ReadVariableOp?,linear_block_2/dense_3/MatMul/ReadVariableOp??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?=linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp??linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1??linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2?Alinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp?-linear_block_3/dense_4/BiasAdd/ReadVariableOp?,linear_block_3/dense_4/MatMul/ReadVariableOp??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
,linear_block_2/dense_3/MatMul/ReadVariableOpReadVariableOp5linear_block_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
linear_block_2/dense_3/MatMulMatMulinputs4linear_block_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-linear_block_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp6linear_block_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
linear_block_2/dense_3/BiasAddBiasAdd'linear_block_2/dense_3/MatMul:product:05linear_block_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
=linear_block_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpFlinear_block_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0y
4linear_block_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
2linear_block_2/batch_normalization_2/batchnorm/addAddV2Elinear_block_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0=linear_block_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
4linear_block_2/batch_normalization_2/batchnorm/RsqrtRsqrt6linear_block_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:?
Alinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpJlinear_block_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
2linear_block_2/batch_normalization_2/batchnorm/mulMul8linear_block_2/batch_normalization_2/batchnorm/Rsqrt:y:0Ilinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
4linear_block_2/batch_normalization_2/batchnorm/mul_1Mul'linear_block_2/dense_3/BiasAdd:output:06linear_block_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
?linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpHlinear_block_2_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
4linear_block_2/batch_normalization_2/batchnorm/mul_2MulGlinear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1:value:06linear_block_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:?
?linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpHlinear_block_2_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
2linear_block_2/batch_normalization_2/batchnorm/subSubGlinear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2:value:08linear_block_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
4linear_block_2/batch_normalization_2/batchnorm/add_1AddV28linear_block_2/batch_normalization_2/batchnorm/mul_1:z:06linear_block_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
 linear_block_2/activation_2/ReluRelu8linear_block_2/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:??????????
!linear_block_2/dropout_2/IdentityIdentity.linear_block_2/activation_2/Relu:activations:0*
T0*'
_output_shapes
:?????????x
	add_1/addAddV2*linear_block_2/dropout_2/Identity:output:0inputs*
T0*'
_output_shapes
:??????????
,linear_block_3/dense_4/MatMul/ReadVariableOpReadVariableOp5linear_block_3_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
linear_block_3/dense_4/MatMulMatMuladd_1/add:z:04linear_block_3/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-linear_block_3/dense_4/BiasAdd/ReadVariableOpReadVariableOp6linear_block_3_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
linear_block_3/dense_4/BiasAddBiasAdd'linear_block_3/dense_4/MatMul:product:05linear_block_3/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
=linear_block_3/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpFlinear_block_3_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0y
4linear_block_3/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
2linear_block_3/batch_normalization_3/batchnorm/addAddV2Elinear_block_3/batch_normalization_3/batchnorm/ReadVariableOp:value:0=linear_block_3/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:?
4linear_block_3/batch_normalization_3/batchnorm/RsqrtRsqrt6linear_block_3/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:?
Alinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpJlinear_block_3_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
2linear_block_3/batch_normalization_3/batchnorm/mulMul8linear_block_3/batch_normalization_3/batchnorm/Rsqrt:y:0Ilinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
4linear_block_3/batch_normalization_3/batchnorm/mul_1Mul'linear_block_3/dense_4/BiasAdd:output:06linear_block_3/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
?linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpHlinear_block_3_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
4linear_block_3/batch_normalization_3/batchnorm/mul_2MulGlinear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1:value:06linear_block_3/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:?
?linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpHlinear_block_3_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
2linear_block_3/batch_normalization_3/batchnorm/subSubGlinear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2:value:08linear_block_3/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
4linear_block_3/batch_normalization_3/batchnorm/add_1AddV28linear_block_3/batch_normalization_3/batchnorm/mul_1:z:06linear_block_3/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:??????????
 linear_block_3/activation_3/ReluRelu8linear_block_3/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:??????????
!linear_block_3/dropout_3/IdentityIdentity.linear_block_3/activation_3/Relu:activations:0*
T0*'
_output_shapes
:??????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_5/MatMulMatMul*linear_block_3/dropout_3/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5linear_block_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5linear_block_3_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp>^linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp@^linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1@^linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2B^linear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp.^linear_block_2/dense_3/BiasAdd/ReadVariableOp-^linear_block_2/dense_3/MatMul/ReadVariableOp@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp>^linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp@^linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1@^linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2B^linear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp.^linear_block_3/dense_4/BiasAdd/ReadVariableOp-^linear_block_3/dense_4/MatMul/ReadVariableOp@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2~
=linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp=linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp2?
?linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_1?linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_12?
?linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_2?linear_block_2/batch_normalization_2/batchnorm/ReadVariableOp_22?
Alinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOpAlinear_block_2/batch_normalization_2/batchnorm/mul/ReadVariableOp2^
-linear_block_2/dense_3/BiasAdd/ReadVariableOp-linear_block_2/dense_3/BiasAdd/ReadVariableOp2\
,linear_block_2/dense_3/MatMul/ReadVariableOp,linear_block_2/dense_3/MatMul/ReadVariableOp2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp2~
=linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp=linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp2?
?linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_1?linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_12?
?linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_2?linear_block_3/batch_normalization_3/batchnorm/ReadVariableOp_22?
Alinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOpAlinear_block_3/batch_normalization_3/batchnorm/mul/ReadVariableOp2^
-linear_block_3/dense_4/BiasAdd/ReadVariableOp-linear_block_3/dense_4/BiasAdd/ReadVariableOp2\
,linear_block_3/dense_4/MatMul/ReadVariableOp,linear_block_3/dense_4/MatMul/ReadVariableOp2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8688057

inputs8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:E
7batch_normalization_2_batchnorm_readvariableop_resource:I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:G
9batch_normalization_2_batchnorm_readvariableop_1_resource:G
9batch_normalization_2_batchnorm_readvariableop_2_resource:
identity??.batch_normalization_2/batchnorm/ReadVariableOp?0batch_normalization_2/batchnorm/ReadVariableOp_1?0batch_normalization_2/batchnorm/ReadVariableOp_2?2batch_normalization_2/batchnorm/mul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp??linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_2/batchnorm/mul_1Muldense_3/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0?
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:?
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0?
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????v
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????q
dropout_2/IdentityIdentityactivation_2/Relu:activations:0*
T0*'
_output_shapes
:??????????
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_2/dense_3/kernel/Regularizer/SquareSquareGlinear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_2/dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_2/dense_3/kernel/Regularizer/SumSum4linear_block_2/dense_3/kernel/Regularizer/Square:y:08linear_block_2/dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_2/dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_2/dense_3/kernel/Regularizer/mulMul8linear_block_2/dense_3/kernel/Regularizer/mul/x:output:06linear_block_2/dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentitydropout_2/Identity:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp@^linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2?
?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp?linear_block_2/dense_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_2_layer_call_fn_8688284

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8686942o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_8687739
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_8686918o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
0__inference_linear_block_3_layer_call_fn_8688156

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8687339o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?P
?
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8688245

inputs8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:K
=batch_normalization_3_assignmovingavg_readvariableop_resource:M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:E
7batch_normalization_3_batchnorm_readvariableop_resource:
identity??%batch_normalization_3/AssignMovingAvg?4batch_normalization_3/AssignMovingAvg/ReadVariableOp?'batch_normalization_3/AssignMovingAvg_1?6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_3/batchnorm/ReadVariableOp?2batch_normalization_3/batchnorm/mul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp??linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_3/moments/meanMeandense_4/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:?
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_4/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:??????????
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(?
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ?
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 p
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0?
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:?
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:?
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0?
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:?
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:?
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:|
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:?
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:?
%batch_normalization_3/batchnorm/mul_1Muldense_4/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:??????????
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:?
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:?
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????v
activation_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_3/dropout/MulMulactivation_3/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????f
dropout_3/dropout/ShapeShapeactivation_3/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*

seed{e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
0linear_block_3/dense_4/kernel/Regularizer/SquareSquareGlinear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:?
/linear_block_3/dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
-linear_block_3/dense_4/kernel/Regularizer/SumSum4linear_block_3/dense_4/kernel/Regularizer/Square:y:08linear_block_3/dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/linear_block_3/dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
-linear_block_3/dense_4/kernel/Regularizer/mulMul8linear_block_3/dense_4/kernel/Regularizer/mul/x:output:06linear_block_3/dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentitydropout_3/dropout/Mul_1:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp@^linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2?
?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp?linear_block_3/dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_2_layer_call_fn_8688297

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8686989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_20
serving_default_input_2:0?????????;
dense_50
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

linear
bn
relu
dropout
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

linear
bn
relu
dropout
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
&iter

'beta_1

(beta_2
	)decay
*learning_rate m?!m?+m?,m?-m?.m?1m?2m?3m?4m? v?!v?+v?,v?-v?.v?1v?2v?3v?4v?"
	optimizer
?
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
 12
!13"
trackable_list_wrapper
f
+0
,1
-2
.3
14
25
36
47
 8
!9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
	regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

+kernel
,bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
@axis
	-gamma
.beta
/moving_mean
0moving_variance
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
+0
,1
-2
.3
/4
05"
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

1kernel
2bias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
[axis
	3gamma
4beta
5moving_mean
6moving_variance
\	variables
]trainable_variables
^regularization_losses
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
`	variables
atrainable_variables
bregularization_losses
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
d	variables
etrainable_variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
10
21
32
43
54
65"
trackable_list_wrapper
<
10
21
32
43"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_5/kernel
:2dense_5/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
"	variables
#trainable_variables
$regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-2linear_block_2/dense_3/kernel
):'2linear_block_2/dense_3/bias
8:62*linear_block_2/batch_normalization_2/gamma
7:52)linear_block_2/batch_normalization_2/beta
@:> (20linear_block_2/batch_normalization_2/moving_mean
D:B (24linear_block_2/batch_normalization_2/moving_variance
/:-2linear_block_3/dense_4/kernel
):'2linear_block_3/dense_4/bias
8:62*linear_block_3/batch_normalization_3/gamma
7:52)linear_block_3/batch_normalization_3/beta
@:> (20linear_block_3/batch_normalization_3/moving_mean
D:B (24linear_block_3/batch_normalization_3/moving_variance
<
/0
01
52
63"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
<	variables
=trainable_variables
>regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
/0
01"
trackable_list_wrapper
<
0
1
2
3"
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
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
<
0
1
2
3"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
/0
01"
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
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
50
61"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 :2dense_5/kernel/m
:2dense_5/bias/m
/:-2linear_block_2/dense_3/kernel/m
):'2linear_block_2/dense_3/bias/m
8:62,linear_block_2/batch_normalization_2/gamma/m
7:52+linear_block_2/batch_normalization_2/beta/m
/:-2linear_block_3/dense_4/kernel/m
):'2linear_block_3/dense_4/bias/m
8:62,linear_block_3/batch_normalization_3/gamma/m
7:52+linear_block_3/batch_normalization_3/beta/m
 :2dense_5/kernel/v
:2dense_5/bias/v
/:-2linear_block_2/dense_3/kernel/v
):'2linear_block_2/dense_3/bias/v
8:62,linear_block_2/batch_normalization_2/gamma/v
7:52+linear_block_2/batch_normalization_2/beta/v
/:-2linear_block_3/dense_4/kernel/v
):'2linear_block_3/dense_4/bias/v
8:62,linear_block_3/batch_normalization_3/gamma/v
7:52+linear_block_3/batch_normalization_3/beta/v
?2?
)__inference_model_1_layer_call_fn_8687254
)__inference_model_1_layer_call_fn_8687772
)__inference_model_1_layer_call_fn_8687805
)__inference_model_1_layer_call_fn_8687590?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_model_1_layer_call_and_return_conditional_losses_8687877
D__inference_model_1_layer_call_and_return_conditional_losses_8687983
D__inference_model_1_layer_call_and_return_conditional_losses_8687638
D__inference_model_1_layer_call_and_return_conditional_losses_8687686?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_8686918input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_linear_block_2_layer_call_fn_8688006
0__inference_linear_block_2_layer_call_fn_8688023?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8688057
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8688104?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_add_1_layer_call_fn_8688110?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_add_1_layer_call_and_return_conditional_losses_8688116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_linear_block_3_layer_call_fn_8688139
0__inference_linear_block_3_layer_call_fn_8688156?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8688190
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8688245?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_5_layer_call_fn_8688254?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_5_layer_call_and_return_conditional_losses_8688265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_8687739input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_2_layer_call_fn_8688284
7__inference_batch_normalization_2_layer_call_fn_8688297?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8688317
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8688351?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_0_8688362?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_3_layer_call_fn_8688381
7__inference_batch_normalization_3_layer_call_fn_8688394?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8688414
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8688448?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_1_8688459?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? ?
"__inference__wrapped_model_8686918u+,0-/.126354 !0?-
&?#
!?
input_2?????????
? "1?.
,
dense_5!?
dense_5??????????
B__inference_add_1_layer_call_and_return_conditional_losses_8688116?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
'__inference_add_1_layer_call_fn_8688110vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8688317b0-/.3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8688351b/0-.3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
7__inference_batch_normalization_2_layer_call_fn_8688284U0-/.3?0
)?&
 ?
inputs?????????
p 
? "???????????
7__inference_batch_normalization_2_layer_call_fn_8688297U/0-.3?0
)?&
 ?
inputs?????????
p
? "???????????
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8688414b63543?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8688448b56343?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
7__inference_batch_normalization_3_layer_call_fn_8688381U63543?0
)?&
 ?
inputs?????????
p 
? "???????????
7__inference_batch_normalization_3_layer_call_fn_8688394U56343?0
)?&
 ?
inputs?????????
p
? "???????????
D__inference_dense_5_layer_call_and_return_conditional_losses_8688265\ !/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_5_layer_call_fn_8688254O !/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8688057d+,0-/.3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
K__inference_linear_block_2_layer_call_and_return_conditional_losses_8688104d+,/0-.3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
0__inference_linear_block_2_layer_call_fn_8688006W+,0-/.3?0
)?&
 ?
inputs?????????
p 
? "???????????
0__inference_linear_block_2_layer_call_fn_8688023W+,/0-.3?0
)?&
 ?
inputs?????????
p
? "???????????
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8688190d1263543?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
K__inference_linear_block_3_layer_call_and_return_conditional_losses_8688245d1256343?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
0__inference_linear_block_3_layer_call_fn_8688139W1263543?0
)?&
 ?
inputs?????????
p 
? "???????????
0__inference_linear_block_3_layer_call_fn_8688156W1256343?0
)?&
 ?
inputs?????????
p
? "??????????<
__inference_loss_fn_0_8688362+?

? 
? "? <
__inference_loss_fn_1_86884591?

? 
? "? ?
D__inference_model_1_layer_call_and_return_conditional_losses_8687638q+,0-/.126354 !8?5
.?+
!?
input_2?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_1_layer_call_and_return_conditional_losses_8687686q+,/0-.125634 !8?5
.?+
!?
input_2?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_model_1_layer_call_and_return_conditional_losses_8687877p+,0-/.126354 !7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_1_layer_call_and_return_conditional_losses_8687983p+,/0-.125634 !7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
)__inference_model_1_layer_call_fn_8687254d+,0-/.126354 !8?5
.?+
!?
input_2?????????
p 

 
? "???????????
)__inference_model_1_layer_call_fn_8687590d+,/0-.125634 !8?5
.?+
!?
input_2?????????
p

 
? "???????????
)__inference_model_1_layer_call_fn_8687772c+,0-/.126354 !7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
)__inference_model_1_layer_call_fn_8687805c+,/0-.125634 !7?4
-?*
 ?
inputs?????????
p

 
? "???????????
%__inference_signature_wrapper_8687739?+,0-/.126354 !;?8
? 
1?.
,
input_2!?
input_2?????????"1?.
,
dense_5!?
dense_5?????????