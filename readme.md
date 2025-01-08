docker run --interactive            --user $(id -u):$(id -g)            --volume "$(pwd):/kubric"   --volume "/nfs/data/banyuanhao:/nfs/data/banyuanhao"         kubricdockerhub/kubruntu   bash ./shell_12.sh

docker run --detach --interactive --user $(id -u):$(id -g) --volume "$(pwd):/kubric" --volume "/nfs/data/banyuanhao:/nfs/data/banyuanhao" kubricdockerhub/kubruntu bash ./shell_program_12.sh

8ba66542e29dc0e763cb36029350602d9babcb93429e46176a0fa45476f7a674
c74d0f9b33b825f8b071a2672e7e9f92d11e53a3b635203c20e1e452d8b1a8d1
633f03f0df6000efffa07507faf15125493a8243c2a6286f2749852046b92014
c0acddf764f0d1cc6e8583c34ca66386dfa7fb86a7b269141a4ee9cd264ba24e