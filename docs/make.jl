using EquivariantModels
using Documenter

DocMeta.setdocmeta!(EquivariantModels, :DocTestSetup, :(using EquivariantModels); recursive=true)

makedocs(;
    modules=[EquivariantModels],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/EquivariantModels.jl/blob/{commit}{path}#{line}",
    sitename="EquivariantModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/EquivariantModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/EquivariantModels.jl",
    devbranch="main",
)
