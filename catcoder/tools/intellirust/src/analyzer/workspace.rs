use std::{collections::HashSet, path::Path};

use anyhow::Result;
use build_target::target_triple;
use cfg::{CfgAtom, CfgDiff};
use hir::Crate;
use ide::{AnalysisHost, FilePosition, NavigationTarget, StructureNode};
use ide_db::{FxHashMap, RootDatabase};
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice};
use paths::AbsPathBuf;
use project_model::{
    CargoConfig, CargoFeatures, CargoWorkspace, CfgOverrides, InvocationLocation,
    InvocationStrategy, Package, PackageData, ProjectManifest, ProjectWorkspace, RustLibSource,
    Target, TargetData, TargetKind,
};
use vfs::{FileId, Vfs, VfsPath};

use crate::__private::*;

pub struct Workspace {
    pub krate: Crate,
    pub host: AnalysisHost,
    pub vfs: Vfs,
}

impl From<&Path> for Workspace {
    fn from(path: &Path) -> Self {
        let (krate, host, vfs) = load_workspace(path).unwrap();
        Self { krate, host, vfs }
    }
}

impl From<&str> for Workspace {
    fn from(path: &str) -> Self {
        Path::new(path).into()
    }
}

type FileStructure = Vec<StructureNode>;
type FileStructures = Vec<FileStructure>;
type Navigations = Vec<NavigationTarget>;

impl Workspace {
    pub fn new(path: &Path) -> Self {
        path.into()
    }

    pub fn file_id(&self, file_path: &Path) -> Option<FileId> {
        self.vfs.file_id(&VfsPath::from(AbsPathBuf::assert(
            file_path.canonicalize().unwrap(),
        )))
    }

    pub fn file_path(&self, file_id: FileId) -> String {
        self.vfs.file_path(file_id).to_string()
    }

    pub fn file_text(&self, file_path: &Path) -> Option<String> {
        self.file_id(file_path)
            .map(|id| self.host.analysis().file_text(id).unwrap().to_string())
    }

    pub fn file_structure(&self, file_path: &Path) -> Option<FileStructure> {
        self.file_id(file_path)
            .map(|id| self.host.analysis().file_structure(id).unwrap())
    }

    pub fn file_structure_by_id(&self, file_id: FileId) -> FileStructure {
        self.host.analysis().file_structure(file_id).unwrap()
    }

    pub fn goto_type_definition(&self, file_path: &Path, offset: usize) -> Option<Navigations> {
        let info = self.file_id(file_path).and_then(|id| {
            self.host
                .analysis()
                .goto_type_definition(FilePosition {
                    file_id: id,
                    offset: offset.try_into().unwrap(),
                })
                .unwrap()
        });
        info.map(|i| i.info)
    }

    pub fn goto_impl_sources(&self, file_path: &Path, offset: usize) -> Option<FileStructures> {
        let info = self.file_id(file_path).and_then(|id| {
            self.host
                .analysis()
                .goto_implementation(FilePosition {
                    file_id: id,
                    offset: offset.try_into().unwrap(),
                })
                .unwrap()
        })?;
        let file_ids = info
            .info
            .iter()
            .map(|nav| nav.file_id)
            .collect::<HashSet<_>>();
        let file_structures = file_ids
            .iter()
            .map(|id| self.file_structure_by_id(*id))
            .collect();
        Some(file_structures)
    }
}

fn load_workspace(path: &Path) -> Result<(Crate, AnalysisHost, Vfs)> {
    let cargo_config = get_cargo_config();
    let load_config = get_load_cargo_config();
    // default progress function, do nothing
    let progress = |_: String| {};
    let mut project_workspace = load_project_workspace(path, &cargo_config, &progress)?;
    let (_package, target) = select_package_and_target(&project_workspace)?;
    if load_config.load_out_dirs_from_check {
        let build_scripts = project_workspace.run_build_scripts(&cargo_config, &progress)?;
        project_workspace.set_build_scripts(build_scripts);
    }
    let (host, vfs, _proc_macro_client) =
        load_cargo::load_workspace(project_workspace, &cargo_config.extra_env, &load_config)?;
    let db = host.raw_database();
    let krate = find_crate(db, &vfs, &target)?;
    Ok((krate, host, vfs))
}

fn load_project_workspace(
    path: &Path,
    config: &CargoConfig,
    progress: &dyn Fn(String),
) -> Result<ProjectWorkspace> {
    let root = AbsPathBuf::assert(path.canonicalize()?);
    let manifest = ProjectManifest::discover_single(root.as_path())?;
    ProjectWorkspace::load(manifest, config, progress)
}

fn get_load_cargo_config() -> LoadCargoConfig {
    LoadCargoConfig {
        load_out_dirs_from_check: false,
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,
        prefill_caches: false,
    }
}

fn get_cargo_config() -> CargoConfig {
    let features = CargoFeatures::All;
    // Use the default target triple for the current host
    let target = match target_triple() {
        Ok(target) => Some(target),
        Err(_) => None,
    };
    let sysroot = Some(RustLibSource::Discover);
    let sysroot_query_metadata = false;
    let sysroot_src = None;
    let rustc_source = None;
    let cfg_overrides = CfgOverrides {
        global: CfgDiff::new(Vec::new(), vec![CfgAtom::Flag("test".into())]).unwrap(),
        selective: Default::default(),
    };
    let wrap_rustc_in_build_scripts = true;
    let run_build_script_command = None;
    let extra_args = vec![];
    let extra_env = FxHashMap::default();
    let invocation_strategy = InvocationStrategy::PerWorkspace;
    let invocation_location = InvocationLocation::Workspace;
    let target_dir = None;

    CargoConfig {
        features,
        target,
        sysroot,
        sysroot_query_metadata,
        sysroot_src,
        rustc_source,
        cfg_overrides,
        wrap_rustc_in_build_scripts,
        run_build_script_command,
        extra_args,
        extra_env,
        invocation_strategy,
        invocation_location,
        target_dir,
    }
}

fn select_package(workspace: &CargoWorkspace) -> Result<Package> {
    let packages = workspace
        .packages()
        .filter(|idx| workspace[*idx].is_member)
        .collect::<Vec<_>>();
    match packages.len() {
        0 => anyhow::bail!("no packages in workspace"),
        1 => Ok(packages[0]),
        _ => anyhow::bail!("multiple packages in workspace, not supported yet"),
    }
}

fn select_target(workspace: &CargoWorkspace, idx: Package) -> Result<Target> {
    let package = &workspace[idx];
    // We only analyze library targets
    // Bin, Example, Test, Bench, BuildScript and Other are ignored
    let targets = package
        .targets
        .iter()
        .cloned()
        .filter(|idx| matches!(&workspace[*idx].kind, TargetKind::Lib))
        .collect::<Vec<_>>();
    match targets.len() {
        0 => anyhow::bail!("no targets in package"),
        1 => Ok(targets[0]),
        _ => anyhow::bail!("multiple lib targets in package, this should not happen"),
    }
}

fn select_package_and_target(workspace: &ProjectWorkspace) -> Result<(PackageData, TargetData)> {
    let cargo_workspace = match workspace {
        ProjectWorkspace::Cargo { cargo, .. } => cargo,
        _ => panic!("not a cargo workspace"),
    };
    let package_idx = select_package(cargo_workspace)?;
    let package = cargo_workspace[package_idx].clone();

    let target_idx = select_target(cargo_workspace, package_idx)?;
    let target = cargo_workspace[target_idx].clone();

    Ok((package, target))
}

fn find_crate(db: &RootDatabase, vfs: &Vfs, target: &TargetData) -> Result<Crate> {
    let crates = Crate::all(db);
    let target_root_path = target.root.as_path();
    let krate = crates.into_iter().find(|krate| {
        let vfs_path = vfs.file_path(krate.root_file(db));
        let crate_root_path = vfs_path.as_path().unwrap();
        crate_root_path == target_root_path
    });
    krate.ok_or_else(|| anyhow::anyhow!("crate not found"))
}
