using Images: imfilter, mapwindow, adjust_histogram!, LinearStretching, Gray, N0f16, N0f8,
              Kernel, warp, axes, KernelFactors, RGB 
using ImageMorphology: label_components, component_lengths
using StatsBase: mean, median, quantile
using TiffImages: load, save
using NaturalSort: sort, natural
using DataFrames: DataFrame
using CSV: write
using JSON: parsefile
using IntegralArrays: IntegralArray
using CoordinateTransformations: Translation
using IntervalSets: width, leftendpoint, rightendpoint, Interval, ±
using SubpixelRegistration: phase_offset
using FLoops

round_odd(x) = div(x, 2) * 2 + 1
compmax(x) = length(x) > 1 ? maximum(x[1:end]) : 0

function read_images(directory, file)
    arr = load("$directory/$file")
    return arr
end

function dust_correct!(masks)
    rows, cols, nframes = size(masks)
    for col in 1:cols
        for row in 1:rows
            if masks[row, col, 1]  
                should_suppress = false
                for t in 2:nframes
                    if !masks[row, col, t]
                        should_suppress = true
                        break
                    end
                end
                if should_suppress
					masks[row, col, :] .= false
                end
            end
        end
    end
    return nothing
end

function normalize_local_contrast(img, img_copy, blockDiameter)
	img = 1 .- img
	img_copy = 1 .- img_copy
	img_copy = imfilter(img_copy, Kernel.gaussian(blockDiameter))
	img = img - img_copy
    return img 
end

function crop(img_stack)
    @views mask = .!any(isnan.(img_stack), dims=3)[:,:,1]
    @views mask_i = any(mask, dims=2)[:,1]
    @views mask_j = any(mask, dims=1)[1,:]
    i1 = findfirst(mask_i)
    i2 = findlast(mask_i)
    j1 = findfirst(mask_j)
    j2 = findlast(mask_j)
    cropped_stack = img_stack[i1:i2, j1:j2, :]
    return cropped_stack, (i1, i2, j1, j2)
end

function stack_preprocess(img_stack, normalized_stack, registered_stack, blockDiameter, nframes, mxshift, sig)       
    shifts = (0.0, 0.0) 
    @inbounds for t in 1:nframes
        img = img_stack[:,:,t]
        img_copy = img_stack[:,:,t] 
        img_normalized = normalize_local_contrast(img, img_copy, 
                                    blockDiameter)
        normalized_stack[:,:,t] = imfilter(img_normalized, Kernel.gaussian(sig))
        if t == 1
            registered_stack[:,:,t] = normalized_stack[:,:,t]
        else
            moving = normalized_stack[:,:,t]
            fixed = normalized_stack[:,:,t-1]
            shift, _, _ = phase_offset(fixed, moving, upsample_factor=1)
            if sqrt(shift[1]^2 + shift[2]^2) >= mxshift
                shift = Translation(shifts[1], shifts[2])
                registered_stack[:,:,t] = warp(moving, shift, axes(fixed))
                img_stack[:,:,t] = warp(img_stack[:,:,t], shift, axes(fixed))
            else
                shift = Tuple([-1*shift[1], -1*shift[2]])
                shift = shift .+ shifts
                shifts = shift
                shift = Translation(shift[1], shift[2])
                registered_stack[:,:,t] = warp(moving, shift, axes(fixed))
                img_stack[:,:,t] = warp(img_stack[:,:,t], shift, axes(fixed))
            end
        end
    end
    processed_stack, crop_indices = crop(registered_stack)
    row_min, row_max, col_min, col_max = crop_indices
    img_stack = img_stack[row_min:row_max, col_min:col_max, :]
    return img_stack, processed_stack
end

function compute_mask!(stack, masks, fixed_thresh, ntimepoints)
    @inbounds for t in 1:ntimepoints
		@views masks[:,:,t] = stack[:,:,t] .> fixed_thresh
    end
end

function output_images!(stack, masks, overlay, dir, file)
	flat_stack = vec(stack)
    img_min = quantile(flat_stack, 0.0035)
    img_max = quantile(flat_stack, 0.9965)
    adjust_histogram!(stack, LinearStretching(src_minval=img_min, src_maxval=img_max, 
                                              dst_minval=0, dst_maxval=1))
	stack = Gray{N0f8}.(stack)
    save("$dir/results_images/$file.tif", stack)
    @inbounds for i in CartesianIndices(stack)
        gray_val = RGB{N0f8}(stack[i], stack[i], stack[i])
        overlay[i] = masks[i] ? RGB{N0f8}(1, 0, 0) : gray_val
    end
    save("$dir/results_images/$file"*"mask.tif", overlay)
end

function main()
    config = parsefile("experiment_config.json")
    images_directories  = config["images_directory"]
    sig = config["sig"]
    blockDiameter = config["blockDiameter"] 
    shift_thresh = config["shift_thresh"]
    dust_correction = config["dust_correction"]
    dir = "raw"
    if isdir("$dir/results_images")
        rm("$dir/results_images"; recursive = true)
    end
    if isdir("$dir/results_data")
        rm("$dir/results_data"; recursive = true)
    end
    mkdir("$dir/results_images")
    mkdir("$dir/results_data")
    BF_output_file = "$dir/results_data/BF_imaging.csv"
    height, width, ntimepoints = size(Array{Int, 3}(undef, 832, 1128, 31))
    files = [f for f in readdir(dir) if occursin(r"\.tif$", f)]
    for file in files

        BF_data_matrix = Array{Float64, 1}(undef, ntimepoints)

        images = read_images(dir, file)
        images = Float64.(images)
        normalized_stack = similar(images)
        registered_stack = similar(images)
        fixed_thresh = 0.03
        images, output_stack = stack_preprocess(images, normalized_stack, 
                                                                        registered_stack, blockDiameter[2], 
                                                                        ntimepoints, shift_thresh, 
                                                                        sig)
        masks = zeros(Bool, size(images))
        compute_mask!(output_stack, masks, fixed_thresh, ntimepoints)
        if dust_correction == "True"
            dust_correct!(masks)
        end
        overlay = zeros(RGB{N0f8}, size(output_stack)...)
        output_images!(output_stack, masks, overlay, dir, file[1:end-4])


        BF_df = Array{Float64}(undef, 0, 31)
        let images = images
            @floop for t in 1:ntimepoints
                @inbounds signal = @views mean((1 .- images[:,:,t]) .* masks[:,:,t])
                @inbounds BF_data_matrix[t] = signal
            end
        BF_df = vcat(BF_df, BF_data_matrix')
        end
    df = DataFrame(BF_df, :auto)
    df .= ifelse.(isnan.(df), 0, df)
    write(BF_output_file, df, append = true)
    end
end
main()