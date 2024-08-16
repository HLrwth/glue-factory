#!/bin/bash

# Usage:
#      source ./setCUDAenv.sh [<CUDA_ver>] [<cuDNN_ver>]
# will set the CUDA environment to specified versions. The versions must match those given in the directory names
# in /is/software/nvidia exactly.
# Without arguments, the script leaves the current cuda environment set or sets the most recent version if no
# CUDA env is set up so far.
# If the current PATH and LD_LIBRARY_PATH setup is inconsistent, the script prints an error and exits.

cleanup() {
	#cleanup local variables
	unset CUDABASEPATH
	unset CUDNNBASEPATH
	unset cudas
	unset cudnns
	unset ver
	unset currCudaVer
	unset currLDCudaVer
	unset currLDCupTiVer
	unset dnnver
	unset currCuDNNVer
	unset currCuDNNLDVer
	unset currCuDNNCPLUSVer
	unset currCuDNNCVer
	unset currCuDNNCudaVer
	unset oldDNNVer
	unset setCudaVer
}

CUDABASEPATH=/is/software/nvidia/cuda-
CUDNNBASEPATH=/is/software/nvidia/cudnn-

# Store available CUDA and cuDNN versions
cudas=$(echo $CUDABASEPATH* | sed "s,$CUDABASEPATH,,g" | sed 's/ /\n/g' | sort -g)
cudnns=$(echo $CUDNNBASEPATH* | sed "s,$CUDNNBASEPATH,,g" | sed 's/ /\n/g' | sort -g)

# Check if one of the CUDA versions is already in the paths (if multiple are there, throw an error).
for ver in $cudas; do
	if [[ ":$PATH:" == *":$CUDABASEPATH$ver/bin:"* ]]; then
		if [ ! -z "$currCudaVer" ]; then
			echo "error: Multiple cuda versions found in PATH: $PATH"
			cleanup
			return 1
			exit 1
		else 	
			echo "Cuda $ver found in PATH: $PATH"
			currCudaVer=$ver
		fi
	fi
	if [[ ":$LD_LIBRARY_PATH:" == *":$CUDABASEPATH$ver/lib64:"* ]]; then
		if [ ! -z "$currLDCudaVer" ]; then
			echo "error: Multiple cuda versions found in LD_LIBRAY_PATH: $LD_LIBRARY_PATH"
			cleanup
			return 1
			exit 1
		else 	
			echo "Cuda $ver found in LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
			currLDCudaVer=$ver
		fi
	fi
	if [[ ":$LD_LIBRARY_PATH:" == *":$CUDABASEPATH$ver/extras/CUPTI/lib64:"* ]]; then
		if [ ! -z "$currLDCupTiVer" ]; then
			echo "error: Multiple cupti versions found in LD_LIBRAY_PATH: $LD_LIBRARY_PATH"
			cleanup
			return 1
			exit 1
		else
			echo "Cupti $ver found in LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
			currLDCupTiVer=$ver
		fi
	fi
done

# Check if a compatible cuDNN version is already set up in the PATH.
for dnnver in $cudnns; do
	if [[ ":$LD_LIBRARY_PATH:" == *":$CUDNNBASEPATH$dnnver/lib64:"* ]]; then
		if [ ! -z "$currCuDNNLDVer" ]; then
			echo "error: Multiple cudnn versions found in LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
			cleanup
			return 1
			exit 1
		else
			oldDNNVer=$dnnver

			# cuDNN versions <= 5.1 only work with specifig CUDA versions, afterwards they carry
			# compatibility information in their folder name.
			case $dnnver in
				2.0)
					currCuDNNCudaVer=6.5
					currCuDNNLDVer=2.0
					;;
				3.0)
					currCuDNNCudaVer=7.0
					currCuDNNLDVer=3.0
					;;
				4.0)
					currCuDNNCudaVer=7.0
					currCuDNNLDVer=4.0
					;;
				5.1)
					currCuDNNCudaVer=7.5
					currCuDNNLDVer=5.1
					;;
				*-cu*)
					currCuDNNCudaVer=$(echo $dnnver | sed 's/^.*-cu//g')
					currCuDNNLDVer=$(echo $dnnver | sed 's/-cu.*//g')
					;;
			esac
			echo "cuDNN $currCuDNNLDVer compatible with cuda $currCuDNNCudaVer found in LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
		fi
	fi
	if [[ ":$LIBRARY_PATH:" == *":$CUDNNBASEPATH$dnnver/lib64:"* ]]; then
		if [ ! -z "$currCuDNNVer" ]; then
			echo "error: Multiple cudnn versions found in LIBRARY_PATH: $LIBRARY_PATH"
			cleanup
			return 1
			exit 1
		else
			oldDNNVer=$dnnver

			# cuDNN versions <= 5.1 only work with specifig CUDA versions, afterwards they carry
			# compatibility information in their folder name.
			case $dnnver in
				2.0)
					currCuDNNCudaVer=6.5
					currCuDNNVer=2.0
					;;
				3.0)
					currCuDNNCudaVer=7.0
					currCuDNNVer=3.0
					;;
				4.0)
					currCuDNNCudaVer=7.0
					currCuDNNVer=4.0
					;;
				5.1)
					currCuDNNCudaVer=7.5
					currCuDNNVer=5.1
					;;
				*-cu*)
					currCuDNNCudaVer=$(echo $dnnver | sed 's/^.*-cu//g')
					currCuDNNVer=$(echo $dnnver | sed 's/-cu.*//g')
					;;
			esac
			echo "cuDNN $currCuDNNVer compatible with cuda $currCuDNNCudaVer found in LIBRARY_PATH: $LIBRARY_PATH"
		fi
	fi
	if [[ ":$CPLUS_INCLUDE_PATH:" == *":$CUDNNBASEPATH$dnnver/include:"* ]]; then
		if [ ! -z "$currCuDNNCPLUSVer" ]; then
			echo "error: Multiple cudnn versions found in CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"
			cleanup
			return 1
			exit 1
		else
			oldDNNVer=$dnnver

			# cuDNN versions <= 5.1 only work with specifig CUDA versions, afterwards they carry
			# compatibility information in their folder name.
			case $dnnver in
				2.0)
					currCuDNNCudaVer=6.5
					currCuDNNCPLUSVer=2.0
					;;
				3.0)
					currCuDNNCudaVer=7.0
					currCuDNNCPLUSVer=3.0
					;;
				4.0)
					currCuDNNCudaVer=7.0
					currCuDNNCPLUSVer=4.0
					;;
				5.1)
					currCuDNNCudaVer=7.5
					currCuDNNCPLUSVer=5.1
					;;
				*-cu*)
					currCuDNNCudaVer=$(echo $dnnver | sed 's/^.*-cu//g')
					currCuDNNCPLUSVer=$(echo $dnnver | sed 's/-cu.*//g')
					;;
			esac
			echo "cuDNN $currCuDNNCPLUSVer compatible with cuda $currCuDNNCudaVer found in CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"
		fi
	fi
	if [[ ":$C_INCLUDE_PATH:" == *":$CUDNNBASEPATH$dnnver/include:"* ]]; then
		if [ ! -z "$currCuDNNCVer" ]; then
			echo "error: Multiple cudnn versions found in C_INCLUDE_PATH: $C_INCLUDE_PATH"
			cleanup
			return 1
			exit 1
		else
			oldDNNVer=$dnnver

			# cuDNN versions <= 5.1 only work with specifig CUDA versions, afterwards they carry
			# compatibility information in their folder name.
			case $dnnver in
				2.0)
					currCuDNNCudaVer=6.5
					currCuDNNCVer=2.0
					;;
				3.0)
					currCuDNNCudaVer=7.0
					currCuDNNCVer=3.0
					;;
				4.0)
					currCuDNNCudaVer=7.0
					currCuDNNCVer=4.0
					;;
				5.1)
					currCuDNNCudaVer=7.5
					currCuDNNCVer=5.1
					;;
				*-cu*)
					currCuDNNCudaVer=$(echo $dnnver | sed 's/^.*-cu//g')
					currCuDNNCVer=$(echo $dnnver | sed 's/-cu.*//g')
					;;
			esac
			echo "cuDNN $currCuDNNCVer compatible with cuda $currCuDNNCudaVer found in C_INCLUDE_PATH: $C_INCLUDE_PATH"
		fi
	fi
done

# Check if the current setup is consistent.
currWildCardCudaVer=$(sed "s/\([0-9]\+\)\.\([0-9]\)/\1\.x/" <<< $currLDCudaVer)
if [ "$currCudaVer" != "$currLDCudaVer" ] || ([ ! -z "$currLDCupTiVer" ] && [ "$currLDCupTiVer" != "$currLDCudaVer" ]) \
	|| ([ ! -z "$currCuDNNCudaVer" ] && (([ "$currCuDNNCudaVer" != "$currLDCudaVer" ] && [ "$currCuDNNCudaVer" != "$currWildCardCudaVer" ]) \
						|| [ "$currCuDNNLDVer" != "$currCuDNNVer" ] || [ "$currCuDNNCVer" != "$currCuDNNVer" ] \
						|| [ "$currCuDNNCPLUSVer" != "$currCuDNNVer" ])); then
	echo "error: Conflicting versions in PATHs"
	echo "PATH: $PATH"
	echo "LIBRARY_PATH: $LIBRARY_PATH"
	echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
	echo "C_INCLUDE_PATH: $C_INCLUDE_PATH"
	echo "CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"
	cleanup
	return 1
	exit 1
fi

# If we are given versions, check if they are available
if [ $# != 0 ]; then
	if [[ ,"$(echo $cudas | sed 's/ /,/g')", != *",$1,"* ]]; then
		echo "error: Cuda version $1 not found."
		cleanup
		return 1
		exit 1
	fi
	if [ $# == 2 ]; then
		case $1 in
			6.0)
				echo "error: No cuDNN available for cuda 6.0."
				cleanup
				return 1
				exit 1
				;;
			6.5)
				if [ "$2" != "2.0" ]; then
					echo "error: Only cuDNN 2.0 available for cuda 6.0."
					cleanup
					return 1
					exit 1
				fi
				;;
			7.0)
				if [ "$2" != "3.0" ] && [ "$2" != "4.0" ]; then
					echo "error: Only cuDNN 3.0 and 4.0 available for cuda 7.0."
					cleanup
					return 1
					exit 1
				fi
				;;
			7.5)
				if [ "$2" != "5.1" ] && [ "$2" != "6.0" ]; then
					echo "error: Only cuDNN 5.1 and 6.0 are available for cuda 7.5."
					cleanup
					return 1
					exit 1
				fi
				;;
			*)
				if [[ ,"$(echo $cudnns | sed 's/ /,/g')", != *",$2-cu$1,"* ]] && [[ ,"$(echo $cudnns | sed 's/ /,/g')", != *",$2-cu$(sed "s/\([0-9]\+\)\.\([0-9]\)/\1\.x/g" <<< $1),"* ]]; then
					echo "error: cuDNN $2 not available for cuda $1."
					cleanup
					return 1
					exit 1
				fi
				;;
		esac
	fi
fi

# Set up PATH (append or change version depending on whether it is set already).
if [ -z "$currCudaVer" ]; then
	if [[ $PATH == *"cuda"* ]]; then
		echo "error: Local cuda installation found, might conflict with server installation. Cancelling..."
		cleanup
		return 1
		exit 1
	fi
	if [ $# == 0 ]; then
		echo "Cuda not set in PATH, setting to newest version."
		export PATH="$CUDABASEPATH$ver/bin"${PATH:+:${PATH}}
		setCudaVer=$ver
	else
		echo "Cuda not set in PATH, setting to version $1."
		export PATH="$CUDABASEPATH$1/bin"${PATH:+:${PATH}}
		setCudaVer=$1
	fi
else
	if ([ $# != 0 ] && [ $currCudaVer == $1 ]) || [ $# == 0 ]; then
		echo "Cuda currently set in PATH to version $currCudaVer."
		setCudaVer=$currCudaVer
	else
		echo "Cuda currently set in PATH to version $currCudaVer, changing to $1."
		export PATH=$(echo $PATH | sed "s,$CUDABASEPATH$currCudaVer/bin,$CUDABASEPATH$1/bin,g")
		setCudaVer=$1
	fi
fi

# Set up LD_LIBRARY_PATH (append or change version depending on whether it is set already). 
if [ -z "$currLDCudaVer" ]; then
	if [[ $LD_LIBRARY_PATH == *"cuda"* ]]; then
		echo "error: Local cuda installation found, might conflict with server installation. Cancelling..."
		cleanup
		return 1
		exit 1
	fi
	if [ $# == 0 ]; then
		echo "Cuda not set in LD_LIBRARY_PATH, setting to newest version."
		export LD_LIBRARY_PATH="$CUDABASEPATH$ver/lib64"${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	else
		echo "Cuda not set in LD_LIBRARY_PATH, setting to version $1."
		export LD_LIBRARY_PATH="$CUDABASEPATH$1/lib64"${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	fi
else
	if ([ $# != 0 ] && [ $currLDCudaVer == $1 ]) || [ $# == 0 ]; then
		echo "Cuda currently set in LD_LIBRARY_PATH to version $currLDCudaVer."
	else
		echo "Cuda currently set in LD_LIBRARY_PATH to version $currLDCudaVer, changing to $1."
		export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s,$CUDABASEPATH$currLDCudaVer/lib64,$CUDABASEPATH$1/lib64,g")
	fi
fi

# Update or append CUPTI in LD_LIBRARY_PATH
if [ -z "$currLDCupTiVer" ]; then
	if [ $# == 0 ]; then
		echo "Cupti not set in LD_LIBRARY_PATH, setting to newest version."
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$CUDABASEPATH$ver/extras/CUPTI/lib64"
	else
		echo "Cupti not set in LD_LIBRARY_PATH, setting to version $1."
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$CUDABASEPATH$1/extras/CUPTI/lib64"
	fi
else
	if ([ $# != 0 ] && [ $currLDCupTiVer == $1 ]) || [ $# == 0 ]; then
		echo "Cupti currently set in LD_LIBRARY_PATH to version $currLDCupTiVer."
	else
		echo "Cupti currently set in LD_LIBRARY_PATH to version $currLDCupTiVer, changing to $1."
		export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s,$CUDABASEPATH$currLDCupTiVer/extras/CUPTI/lib64,$CUDABASEPATH$1/extras/CUPTI/lib64,g")
	fi
fi

# Set up cuDNN to a compatible version or the requested one if it is compatible.
if [ -z "$currCuDNNVer" ]; then
	if [[ $# < 2 ]]; then
		echo "cuDNN not set in LD_LIBRARY_PATH, setting to newest version compatible with cuda $setCudaVer."
		case $setCudaVer in
			6.0)
				echo "No cuDNN available for cuda 6.0."
				;;
			6.5)
				echo "Setting cuDNN to version 2.0 for cuda 6.5"
				export C_INCLUDE_PATH="${CUDNNBASEPATH}2.0/include"${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
				export CPLUS_INCLUDE_PATH="${CUDNNBASEPATH}2.0/include"${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
				export LIBRARY_PATH="${CUDNNBASEPATH}2.0/lib64"${LIBRARY_PATH:+:${LIBRARY_PATH}}
				export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${CUDNNBASEPATH}2.0/lib64"
				;;
			7.0)
				echo "Setting cuDNN to version 4.0 for cuda 7.0"
				export C_INCLUDE_PATH="${CUDNNBASEPATH}4.0/include"${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
				export CPLUS_INCLUDE_PATH="${CUDNNBASEPATH}4.0/include"${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
				export LIBRARY_PATH="${CUDNNBASEPATH}4.0/lib64"${LIBRARY_PATH:+:${LIBRARY_PATH}}
				export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${CUDNNBASEPATH}4.0/lib64"
				;;
			*)
				# Prefer exact version over wildcard
				if [[ "$cudnns" == *"cu$setCudaVer"* ]]; then
					setCuDNNVer=$(echo $cudnns | sed "s/^.* \([0-9]\(\.[0-9]\)\+-cu$setCudaVer\).*/\1/")
				else
					wildCardCudaVer=$(sed "s/\([0-9]\+\)\.\([0-9]\)/\1\.x/" <<< $setCudaVer)
					if [[ "$cudnns" == *"cu$wildCardCudaVer"* ]]; then
						setCuDNNVer=$(echo $cudnns | sed "s/^.* \([0-9]\(\.[0-9]\)\+-cu$wildCardCudaVer\).*/\1/")
					else
						echo "error: Could not find compatible cuDNN for cuda $setCUDAVer"
						cleanup
						return 1
						exit 1
					fi
				fi
				echo "Setting cuDNN to version $setCuDNNVer for cuda $setCudaVer"
				;;
		esac
	else
		echo "cuDNN not set in LD_LIBRARY_PATH, setting to version $2."
		case $2 in
			2.0|3.0|4.0|5.1)
				export C_INCLUDE_PATH="${CUDNNBASEPATH}$2/include"${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
				export CPLUS_INCLUDE_PATH="${CUDNNBASEPATH}$2/include"${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
				export LIBRARY_PATH="${CUDNNBASEPATH}$2/lib64"${LIBRARY_PATH:+:${LIBRARY_PATH}}
				export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$CUDNNBASEPATH$2/lib64"
				;;
			*)
				# Prefer exact version over wildcard
				if [[ "$cudnns" == *"$2-cu$setCudaVer"* ]]; then
					setCuDNNVer="$2-cu$setCudaVer"
				else
					wildCardCudaVer=$(sed "s/\([0-9]\+\)\.\([0-9]\)/\1\.x/" <<< $setCudaVer)
					setCuDNNVer="$2-cu$wildCardCudaVer"
				fi
				;;
		esac
	fi
	export C_INCLUDE_PATH="${CUDNNBASEPATH}$setCuDNNVer/include"${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}
	export CPLUS_INCLUDE_PATH="${CUDNNBASEPATH}$setCuDNNVer/include"${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
	export LIBRARY_PATH="${CUDNNBASEPATH}$setCuDNNVer/lib64"${LIBRARY_PATH:+:${LIBRARY_PATH}}
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$CUDNNBASEPATH$setCuDNNVer/lib64"
else
	if [ $# == 0 ] || ([ $# == 1 ] && [ $currLDCudaVer == $1 ]); then
		echo "cuDNN currently set to $currCuDNNVer for cuda $currCudaVer."
	elif [ $# == 1 ]; then
		echo "Cuda version changed, setting cuDNN to newest compatible version."
		case $setCudaVer in
			6.0)
				echo "No cuDNN available for cuda 6.0."
				export C_INCLUDE_PATH=$(echo $C_INCLUDE_PATH | sed "s,:$CUDNNBASEPATH$oldDNNVer/include,,g")
				export CPLUS_INCLUDE_PATH=$(echo $CPLUS_INCLUDE_PATH | sed "s,:$CUDNNBASEPATH$oldDNNVer/include,,g")
				export LIBRARY_PATH=$(echo $LIBRARY_PATH | sed "s,:$CUDNNBASEPATH$oldDNNVer/lib64,,g")
				export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s,:$CUDNNBASEPATH$oldDNNVer/lib64,,g")
				;;
			6.5)
				echo "Setting cuDNN to version 2.0 for cuda 6.5"
				export C_INCLUDE_PATH=$(echo $C_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}2.0/include,g")
				export CPLUS_INCLUDE_PATH=$(echo $CPLUS_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}2.0/include,g")
				export LIBRARY_PATH=$(echo $LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,${CUDNNBASEPATH}2.0/lib64,g")
				export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,${CUDNNBASEPATH}2.0/lib64,g")
				;;
			7.0)
				echo "Setting cuDNN to version 4.0 for cuda 7.0"
				export C_INCLUDE_PATH=$(echo $C_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}4.0/include,g")
				export CPLUS_INCLUDE_PATH=$(echo $CPLUS_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}4.0/include,g")
				export LIBRARY_PATH=$(echo $LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,${CUDNNBASEPATH}4.0/lib64,g")
				export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,${CUDNNBASEPATH}4.0/lib64,g")
				;;
			*)
				# Prefer exact match over wildcard
				if [[ "$cudnns" == *"cu$setCudaVer"* ]]; then
					setCuDNNVer=$(echo $cudnns | sed "s/^.* \([0-9]\(\.[0-9]\)\+-cu${setCudaVer}\).*/\1/")
				else
					wildCardCudaVer=$(sed "s/\([0-9]\+\)\.\([0-9]\)/\1\.x/" <<< $setCudaVer)
					if [[ "$cudnns" == *"cu$wildCardCudaVer"* ]]; then
						setCuDNNVer=$(echo $cudnns | sed "s/^.* \([0-9]\(\.[0-9]\)\+-cu$wildCardCudaVer\).*/\1/")
					else
						echo "error: Could not find compatible cuDNN for cuda $setCUDAVer"
						cleanup
						return 1
						exit 1
					fi
				fi
				echo "Setting cuDNN to version $setCuDNNVer for cuda $setCudaVer"
				export C_INCLUDE_PATH=$(echo $C_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}$setCuDNNVer/include,g")
				export CPLUS_INCLUDE_PATH=$(echo $CPLUS_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}$setCuDNNVer/include,g")
				export LIBRARY_PATH=$(echo $LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,${CUDNNBASEPATH}$setCuDNNVer/lib64,g")
				export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,$CUDNNBASEPATH$setCuDNNVer/lib64,g")
				;;
		esac
	else
		echo "Setting cuDNN to version $2 for cuda $1."
		case $2 in
			2.0|3.0|4.0|5.1)
				export C_INCLUDE_PATH=$(echo $C_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}$2/include,g")
				export CPLUS_INCLUDE_PATH=$(echo $CPLUS_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}$2/include,g")
				export LIBRARY_PATH=$(echo $LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,${CUDNNBASEPATH}$2/lib64,g")
				export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,$CUDNNBASEPATH$2/lib64,g")
				;;
			*)
				if [[ "$cudnns" == *"$2-cu$1" ]]; then
					setCuDNNVer=$2-cu$1
				else
					wildCardCudaVer=$(sed "s/\([0-9]\+\)\.\([0-9]\)/\1\.x/" <<< $1)
					if [[ "$cudnns" == *"$2-cu$wildCardCudaVer"* ]]; then
						setCuDNNVer=$2-cu$wildCardCudaVer
					else
						echo "error: Something went wrong, cuDNN version $2 not available for cuda $1"
						cleanup
						return 1
						exit 1
					fi
				fi
				export C_INCLUDE_PATH=$(echo $C_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}$setCuDNNVer/include,g")
				export CPLUS_INCLUDE_PATH=$(echo $CPLUS_INCLUDE_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/include,${CUDNNBASEPATH}$setCuDNNVer/include,g")
				export LIBRARY_PATH=$(echo $LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,${CUDNNBASEPATH}$setCuDNNVer/lib64,g")
				export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s,$CUDNNBASEPATH$oldDNNVer/lib64,$CUDNNBASEPATH$setCuDNNVer/lib64,g")
				;;
		esac
	fi
fi

export CUDA_HOME=$CUDABASEPATH$setCudaVer
export CUDNN_ROOT_DIR=$CUDNNBASEPATH$setCuDNNVer

echo "CUDA_HOME: $CUDA_HOME"
echo "CUDNN_ROOT_DIR: $CUDNN_ROOT_DIR"
echo "PATH: $PATH"
echo "LIBRARY_PATH: $LIBRARY_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "C_INCLUDE_PATH: $C_INCLUDE_PATH"
echo "CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"

cleanup
