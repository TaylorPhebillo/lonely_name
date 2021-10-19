using Base.Sort
using DataStructures
using DataFrames
using StatsBase
using Random
using CSV

function read_data(filename="common_words.vectors")
  return DataFrame(CSV.File(filename, header=0, delim=" "))
end

function read_names(limit, filename="lexvec.commoncrawl.300d.W.pos.vectors")
  return CSV.File(filename, header=0, delim=" ", limit=limit+1).Column1[2:end]
end

function quantize(val)
  return val >0 ? 1 : -1
end

function locality_sensitive_hash(data_m, reduced_dims)
  r = rand(size(data, 2), reduced_dims) .* 2 .- 1
  projected = data_m * r
  return quantize.(projected)
end

# Given an LSH, map from buckets to indices that fall in that bucket. Find the smallest buckets, return their indices, in the hopes that the lonelist word will fall in a sparse bucket
function most_isolated_indices(lsh, limit)
  d = DefaultDict(() -> Int64[])
  for (index, r) in enumerate(eachrow(lsh))
#    print(r, ", ", index)
    push!(d[r], index)
  end
  to_ret = []
  for (hash, indices) in sort(collect(d), by=x->size(x[2]))
    append!(to_ret, indices)
    if size(indices, 1) > limit
      break
    end
  end
  return sort(to_ret)
end

# LSH is probabalistic- repeat many times, collect potentially lonely indices from each new LSH
function many_isolated_indices(data_m, reduced_dims, limit, repeat)
  candidates = []
  for i in 1:repeat
    append!(candidates, most_isolated_indices(locality_sensitive_hash(data_m, reduced_dims), limit))
  end
  return candidates
end


# Euclidian distance, but don't consider a point close to itself
function e_dist(v1, v2)
  d = sum((v1 - v2) .^ 2)
  if d == 0.0
    return Inf
  end
  return d
end
# Find the point in the dataset that has the minimum distance to a reference point (data_m[index])
function dist(index, data_m)
  return minimum(e_dist.(Ref(data_m[index, :]), eachrow(data_m)))
end

# Find the num_candidates closest words to a point in the dataset by computing that points distance to each point, sorting just the first part of the list, and return those distances and the names/words those points correspond to
function min_percentile(index, data_m, num_candidates, names)
  calculated_dist = e_dist.(Ref(data_m[index, :]), eachrow(data_m))
  indices = sort(collect(
			 zip(calculated_dist, 1:size(data_m, 1)))
  ; alg=Sort.PartialQuickSort(num_candidates))[1:num_candidates]
  push!(indices, (0.,index))
  mapped_names = names[last.(indices)]
  
  return collect(zip(calculated_dist[last.(indices)], mapped_names))
  #return indices
end

function main()
  data_m = Matrix(read_data()) # Raw vectors
  names = read_names(size(data_m, 1)) # Words those vectors correspond to, from another file for no good reason
  isolated = sort(many_isolated_indices(data_m, 12, 10, 10)) 
  isolated_distances = dist.(isolated[1:500], Ref(data_m))
  most_isolated = sort(collect(zip(isolated_distances, isolated[1:500])))
  closest_to_isolated_names = min_percentile.(last.(most_isolated[491:500]), Ref(data_m), 10, Ref(names))
  for iso_name in reverse(closest_to_isolated_names)
    println(iso_name[end][2] ," has closest_words ", filter(x->!occursin("0", x) && !occursin("<", x), last.(iso_name[1:end-1])))
  end
end
