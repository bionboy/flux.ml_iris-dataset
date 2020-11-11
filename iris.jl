### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 2ae5de88-04dd-11eb-1738-49529e3ef566
begin
	import Pkg;
	packages = ["CSV","DataFrames","PlutoUI","Plots","Combinatorics"]	
	Pkg.add(packages)
	
	using CSV, DataFrames, PlutoUI, Plots, Combinatorics

	plotly()
	theme(:solarized_light)
end

# ╔═╡ d38fc0ac-05b3-11eb-3ebe-cd4920fa0134
begin
	Pkg.add("Flux")
	Pkg.add("CUDA")
	Pkg.add("IterTools")
	
	using Flux
	using Flux: Data.DataLoader
	using Flux: @epochs
	using CUDA
	using Random
	using IterTools: ncycle
	
	Random.seed!(123);

# 	CUDA.allowscalar(false)
end

# ╔═╡ 616bef00-04ed-11eb-08d6-dbf4766ebfa7
md"""
# The Iris data set
"""

# ╔═╡ 7522bc98-05b2-11eb-05b8-19d021ebded6
md"""
[https://archive.ics.uci.edu/ml/datasets/irisReporton]()
"""

# ╔═╡ 7f5bde2e-04ef-11eb-2b54-d52990955957
md"""
## [1] Data set description and possible applications
"""

# ╔═╡ ae4ca166-04ef-11eb-37e1-71de4c21e37d
md"""
This data set contains 150 samples iris flower. The features in each sample are the length and width of both the iris petal and sepal, and also the species of iris. data = 150×5

Each feature is recorded as a floating point value except for the species (string). The species identifier acts as the labels for this data set (if used for supervised learning).There are no missing values. The data and header is seperated into two different files.

This data could be used for iris classification. This could be useful in an automation task involving these flowers or as a tool for researchers to assist in quick identification. Other, less "real world" applications include use as a data set for ML systems such as supervised learning (NN) and unsupervised learning (K-NN).
"""

# ╔═╡ 9dac1ee0-04ef-11eb-0bc7-15dd0e060c95
md"""
## [2] Data summary and visualizations
"""

# ╔═╡ 85f1e2e4-04ed-11eb-2ca0-35026d3dcce0
md"""
### Imports 
"""

# ╔═╡ 7f31fed8-04f7-11eb-1849-250b87b7b8c2


# ╔═╡ 978e85fe-04ed-11eb-3e3c-2bbbf1d114ec
md"""
### Loading, cleaning, and manipulating the data 
"""

# ╔═╡ 971e8f0a-04e2-11eb-3b37-9bfb7785c087
begin
	path = "iris/iris.data"
	csv_data = CSV.File(path, header=false)
	
	iris_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
	df = DataFrame(csv_data.columns, Symbol.(iris_names))
	dropmissing!(df)
	
	md"""
	**Column names:** $(join(iris_names, ", "))
	$(describe(df, cols=1:4))
	"""
end

# ╔═╡ 82808c1e-04f7-11eb-1bc7-b725cfdaa249


# ╔═╡ 4b2eb2a2-04ef-11eb-2d97-55ca340ff3b8
md"""
#### Splitting the data into three iris classes
As you can see, there is a equal representation of each class:
"""

# ╔═╡ ecde3e7a-04ee-11eb-1851-c585ed70e671
begin
	df_species = groupby(df, :class)
	md"""**Class sizes:** $(size(df_species[1])), $(size(df_species[2])) $(size(df_species[3]))"""
end

# ╔═╡ 796f3dbe-04f7-11eb-17b0-811821d49986


# ╔═╡ b44c1bbe-04ed-11eb-36f6-4bddf8abf408
md"""
### Visualizations
"""

# ╔═╡ bf78e8a0-04ed-11eb-0330-0f3546289f14
md"""
#### Comparing length vs width of the sepal and petal 
"""

# ╔═╡ 63358b10-04da-11eb-2bf5-ff28c8aae03c
begin
	scatter(title="len vs wid", xlabel = "length", ylabel="width",
		     df.sepal_len, df.sepal_wid, color="blue", label="sepal")
	scatter!(df.petal_len, df.petal_wid, color="red", label="petal")
end

# ╔═╡ 3a2e3b8c-2432-11eb-3bf9-8f932e252535


# ╔═╡ ca0d4700-04e5-11eb-082e-83e4c086a8f8
begin
	# Get all combinations of colums
	combins = collect(combinations(1:4,2))
	combos = [(df[x][1], df[x][2]) for x in combins]
	# Plot all combinations in sub-plots
	scatter(combos, layout=(2,3))
end

# ╔═╡ cdc3422a-04ed-11eb-3ea6-6bd8b8a05b8e
md"""
#### Comparing all combinations of variables
**Column pairs per chart:** 
[$(join(iris_names, ", "))]

->	
$(join(combins[1:3], " , "))

->	
$(join(combins[4:6], " , "))
"""

# ╔═╡ 37c870d8-2432-11eb-0331-8d9d447e63a4


# ╔═╡ d8fa88ce-04ed-11eb-16bd-690dd8bcec84
md"""
#### Comparing the sepal length vs sepal width vs petal length of all three classes of iris 
Restricted to three variables to plot in 3d
"""

# ╔═╡ a1d717d2-04e7-11eb-382c-1b24eb7accc5
begin
	setosa, versicolor, virginica = df_species
	
	scatter(setosa[1], setosa[2], setosa[3], label="Setosa", xlabel="d")
	scatter!(versicolor[1], versicolor[2], versicolor[3], label="versicolor")
	scatter!(virginica[1], virginica[2], virginica[3], label="virginica")
end

# ╔═╡ 3f979b9a-2432-11eb-1b3c-b978c4737db9


# ╔═╡ 7aa30672-05b3-11eb-1c0b-c16e0ee56057
md"""
## [3] Deep Learning
"""

# ╔═╡ 82e67cb0-05b3-11eb-218c-9987a23aa47f
md"""
### Imports 
"""

# ╔═╡ 1317cab6-072e-11eb-13df-1f3b73f6985f


# ╔═╡ 8d357e14-0699-11eb-3093-e9af10267090
md""" ### The Data """

# ╔═╡ b9a67250-068a-11eb-39e8-6b843fb61ca6
begin	
	# Convert df to array
	data = convert(Array, df)
	
	# Shuffle
	data = data[shuffle(1:end), :]

	# train/test split
	train_test_ratio = .7
	idx = Int(floor(size(df, 1) * train_test_ratio))
	data_train = data[1:idx,:]
	data_test = data[idx+1:end, :]

	# Get feature vectors
	get_feat(d) = transpose(convert(Array{Float32},d[:, 1:end-1]))
	x_train = get_feat(data_train)
	x_test = get_feat(data_test)
	
	# One hot labels
	# 	onehot(d) = [Flux.onehot(v, unique(df.class)) for v in d[:,end]]
	onehot(d) = Flux.onehotbatch(d[:,end], unique(df.class))
	y_train = onehot(data_train)
	y_test = onehot(data_test)

	# Push data onto the GPU	
# 	x_train = cu(x_train)
# 	x_test = cu(x_test)
# 	y_train = cu(y_train)
# 	y_test = cu(y_test)
	
	md"""
	Formating data for training (including onehot conversion and (NOT) moving to gpu)
	"""
end

# ╔═╡ 4e26ee42-2432-11eb-33d5-8b3e65a7fb6d


# ╔═╡ 975954a8-0692-11eb-0317-37cafc99bd9d
begin
	batch_size= 1
	train_dl = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
	test_dl = DataLoader((x_test, y_test), batchsize=batch_size)
	
	md"""#### Creating DataLoaders for batches"""
end

# ╔═╡ 02c8f920-072e-11eb-355f-edee13697dc7


# ╔═╡ 448522e2-0689-11eb-3267-c9d26925be5d
begin
	### Model ------------------------------
	function get_model()
		c = Chain(
			Dense(4,8,relu),
			Dense(8,3),
			softmax
		)
# 		c = cu(c)
	end
	
	model = get_model()

	### Loss ------------------------------
	loss(x,y) = Flux.Losses.logitbinarycrossentropy(model(x), y)
	
	train_losses = []
	test_losses = []
	train_acces = []
	test_acces = []
	
	### Optimiser ------------------------------
	lr = 0.001
	opt = ADAM(lr, (0.9, 0.999))

	### Callbacks ------------------------------
	function loss_all(data_loader)
		sum([loss(x, y) for (x,y) in data_loader]) / length(data_loader) 
	end
	
	function acc(data_loader)
 		f(x) = Flux.onecold(cpu(x))
		acces = [sum(f(model(x)) .== f(y)) / size(x,2)  for (x,y) in data_loader]
    	sum(acces) / length(data_loader)
	end
	
	callbacks = [
		() -> push!(train_losses, loss_all(train_dl)),
		() -> push!(test_losses, loss_all(test_dl)),
		() -> push!(train_acces, acc(train_dl)),
		() -> push!(test_acces, acc(test_dl)),
	]

	# Training ------------------------------
	epochs = 30
	ps = Flux.params(model)
	
	@epochs epochs Flux.train!(loss, ps, train_dl, opt, cb = callbacks)
	
	@show train_loss = loss_all(train_dl)
	@show test_loss = loss_all(test_dl)
	@show train_acc = acc(train_dl)
	@show test_acc = acc(test_dl)
	
	md"""
	### Training!
	**Train**	
	  
	  acc: $(train_acc)
	  
	  loss: $(train_loss)
	
	**Test**
	
	  acc: $(test_acc)
	  
	  loss: $(test_loss)
	"""
end

# ╔═╡ 467df1da-0686-11eb-2548-43814f5cdb10
begin
	md"""
	### The model
	**I am going to implement a fully connected neural network to classify by species.**
	
	**Layers:** $(model)
	
	**Loss:** logit binary crossentropy
	
	**Optimizer:** $(typeof(opt))
	
	**Learning rate:** $(lr)
	
	**Epochs:** $(epochs)
	
	**Batch size:** $(batch_size)
	
	"""
end

# ╔═╡ fe90e50a-072d-11eb-26c8-bd97227bb097


# ╔═╡ f6207c6c-072d-11eb-14a9-21bf3d1c88ab
md"""
### Results
"""

# ╔═╡ 44fd6e58-0721-11eb-09a9-6780dca8fffa
begin
	x_axis = 1:epochs * size(y_train,2)
	plot(x_axis, train_losses, label="Training loss",
		title="Loss", xaxis="epochs * data size")
	plot!(x_axis, test_losses, label="Testing loss")
end

# ╔═╡ 64ed9362-0723-11eb-1075-b78ca5b40926
begin
	plot(x_axis, train_acces, label="Training acc",
		title="Accuracy", xaxis="epochs * data size")
	plot!(x_axis, test_acces, label="Testing acc")
end

# ╔═╡ ef1bbf32-06a1-11eb-0751-d939704e1d62
begin
	y = (y_test[:,1])
	pred = (model(x_test[:,1]))
	
	md"""
	#### One example prediction:
	Prediction: $(join(pred, " , "))
	
	Truth: $(join(Array{Int}(y), " , "))
	
	error: $(sum(abs.(y-pred)))
	"""
end

# ╔═╡ 2a5b168e-072d-11eb-12f8-134944a56b2f
md"""
#### Confusion matrix
"""

# ╔═╡ 5a039506-06a5-11eb-3f13-2b3bc427a0cd
begin
	preds = round.(model(x_test))
	truths = y_test
	
	un_onehot(v) = v[1] == true ? 1 : v[2] == true ? 2 : 3

	preds = [un_onehot(v) for v in eachcol(preds)]
	truths = [un_onehot(v) for v in eachcol(truths)]
	
	conf_mat = zeros(3,3)
	for (y′, y) in zip(preds, truths)	
		if y == 1
			if y′ == 1
				conf_mat[1,1] += 1
			elseif y′ == 2
				conf_mat[1,2] += 1
			else
				conf_mat[1,3] += 1
			end
		elseif y == 2
			if y′ == 1
				conf_mat[2,1] += 1
			elseif y′ == 2
				conf_mat[2,2] += 1
			else
				conf_mat[2,3] += 1
			end
		else
			if y′ == 1
				conf_mat[3,1] += 1
			elseif y′ == 2
				conf_mat[3,2] += 1
			else
				conf_mat[3,3] += 1
			end
		end
	end

# 	conf_mat = conf_mat ./ sum(conf_mat) # normalize
	label = "setosa \t:\t versicolor \t:\t virginica"
	heatmap(conf_mat, color=:plasma, aspect_ratio=1, xaxis=label, axis = nothing)
	
end

# ╔═╡ 61e36e4a-072c-11eb-047d-3b8c5e94ef44


# ╔═╡ 6fe80c24-072e-11eb-0801-af4edc737b72
md"""
## [4] Conclusion

### Platform/Tools

I chose to implement a basic feed forward neural network because of the scale of the problem. With the data set containing so few samples with very little features a small network would fit better. Again, because of the size of the problem, shallow ML approaches would have been sufficient. Something to expand on in this research is to compare to such methods.

I wanted to challenge myself and learn an entirely new language and platform for this project. [The Julia Programming Language](https://julialang.org/) is a high level, dynamically typed language. It comes with its own web-based editor that is much like Python's [Jupter notebooks](https://jupyter.org/). Because Julia is newer and the community is smaller than Python, the documentation and support were not even close in magnitude. This slowed me down considerably. Despite the setbacks, I learned a lot in this research and I am glad I decided to use Julia.

##### Results

My model's test accuracy was 95.55%. This is satisfactory for me due to the simplicity of the data set and the model. While one species was linearly seperable, the other two were not. These later species are the main problem for the model to tackle.

As I stated in the beginning of this paper, this model could be used for classification tasks such as automation or as a tool for bio researchers to aid in identification. Furthermore, this model could be used as a pre-trained model for more specific tasks; I understand this statement is a bit of a stretch but I want to account for as many applications as possible.
"""

# ╔═╡ a46dde7e-072e-11eb-3209-f5e683f5a835
md"""
## [5] Related work

**Related research:** [Kaggle](https://www.kaggle.com/kamrankausar/iris-dataset-ml-and-deep-learning-from-scratch/notebook)

_One thing they did, that I didn't do, is compare their deep learning model to more classical approaches._

"""

# ╔═╡ f2d8adda-072b-11eb-1630-eff35d2e6805
md"""
## References
1. [The Iris Data-set](https://archive.ics.uci.edu/ml/datasets/iris)
1. [Flux.jl](https://fluxml.ai/Flux.jl/stable/)
1. [Exploring High Level APIs of Knet.jl and Flux.jl in comparison to Tensorflow-Keras](https://estadistika.github.io/julia/python/packages/knet/flux/tensorflow/machine-learning/deep-learning/2019/06/20/Deep-Learning-Exploring-High-Level-APIs-of-Knet.jl-and-Flux.jl-in-comparison-to-Tensorflow-Keras.html)
1. [Related Kaggle work](https://www.kaggle.com/kamrankausar/iris-dataset-ml-and-deep-learning-from-scratch/notebook)
"""

# ╔═╡ Cell order:
# ╟─616bef00-04ed-11eb-08d6-dbf4766ebfa7
# ╟─7522bc98-05b2-11eb-05b8-19d021ebded6
# ╟─7f5bde2e-04ef-11eb-2b54-d52990955957
# ╟─ae4ca166-04ef-11eb-37e1-71de4c21e37d
# ╟─9dac1ee0-04ef-11eb-0bc7-15dd0e060c95
# ╟─85f1e2e4-04ed-11eb-2ca0-35026d3dcce0
# ╠═2ae5de88-04dd-11eb-1738-49529e3ef566
# ╟─7f31fed8-04f7-11eb-1849-250b87b7b8c2
# ╟─978e85fe-04ed-11eb-3e3c-2bbbf1d114ec
# ╠═971e8f0a-04e2-11eb-3b37-9bfb7785c087
# ╟─82808c1e-04f7-11eb-1bc7-b725cfdaa249
# ╟─4b2eb2a2-04ef-11eb-2d97-55ca340ff3b8
# ╠═ecde3e7a-04ee-11eb-1851-c585ed70e671
# ╟─796f3dbe-04f7-11eb-17b0-811821d49986
# ╟─b44c1bbe-04ed-11eb-36f6-4bddf8abf408
# ╟─bf78e8a0-04ed-11eb-0330-0f3546289f14
# ╠═63358b10-04da-11eb-2bf5-ff28c8aae03c
# ╟─3a2e3b8c-2432-11eb-3bf9-8f932e252535
# ╟─cdc3422a-04ed-11eb-3ea6-6bd8b8a05b8e
# ╠═ca0d4700-04e5-11eb-082e-83e4c086a8f8
# ╟─37c870d8-2432-11eb-0331-8d9d447e63a4
# ╟─d8fa88ce-04ed-11eb-16bd-690dd8bcec84
# ╠═a1d717d2-04e7-11eb-382c-1b24eb7accc5
# ╟─3f979b9a-2432-11eb-1b3c-b978c4737db9
# ╟─7aa30672-05b3-11eb-1c0b-c16e0ee56057
# ╟─82e67cb0-05b3-11eb-218c-9987a23aa47f
# ╠═d38fc0ac-05b3-11eb-3ebe-cd4920fa0134
# ╟─1317cab6-072e-11eb-13df-1f3b73f6985f
# ╟─8d357e14-0699-11eb-3093-e9af10267090
# ╠═b9a67250-068a-11eb-39e8-6b843fb61ca6
# ╟─4e26ee42-2432-11eb-33d5-8b3e65a7fb6d
# ╠═975954a8-0692-11eb-0317-37cafc99bd9d
# ╟─02c8f920-072e-11eb-355f-edee13697dc7
# ╟─467df1da-0686-11eb-2548-43814f5cdb10
# ╠═448522e2-0689-11eb-3267-c9d26925be5d
# ╟─fe90e50a-072d-11eb-26c8-bd97227bb097
# ╟─f6207c6c-072d-11eb-14a9-21bf3d1c88ab
# ╠═44fd6e58-0721-11eb-09a9-6780dca8fffa
# ╠═64ed9362-0723-11eb-1075-b78ca5b40926
# ╟─ef1bbf32-06a1-11eb-0751-d939704e1d62
# ╟─2a5b168e-072d-11eb-12f8-134944a56b2f
# ╠═5a039506-06a5-11eb-3f13-2b3bc427a0cd
# ╟─61e36e4a-072c-11eb-047d-3b8c5e94ef44
# ╟─6fe80c24-072e-11eb-0801-af4edc737b72
# ╟─a46dde7e-072e-11eb-3209-f5e683f5a835
# ╟─f2d8adda-072b-11eb-1630-eff35d2e6805
