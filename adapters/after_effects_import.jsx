/*
ParallaxStudio3D After Effects importer.
Choose the generated layers folder containing manifest.json.
*/
(function () {
    var folder = Folder.selectDialog("Choose ParallaxStudio3D layers folder");
    if (!folder) return;
    var manifestFile = new File(folder.fsName + "/manifest.json");
    if (!manifestFile.exists) {
        alert("manifest.json was not found.");
        return;
    }
    manifestFile.open("r");
    var manifest = JSON.parse(manifestFile.read());
    manifestFile.close();

    app.beginUndoGroup("Import ParallaxStudio3D");
    var comp = app.project.items.addComp(
        "ParallaxStudio3D", manifest.width, manifest.height, 1, 6, 24
    );
    var backgroundFile = new File(folder.fsName + "/" + manifest.background);
    var background = app.project.importFile(new ImportOptions(backgroundFile));
    comp.layers.add(background).name = "00 Background";

    for (var i = 0; i < manifest.layers.length; i++) {
        var entry = manifest.layers[i];
        var footage = app.project.importFile(
            new ImportOptions(new File(folder.fsName + "/" + entry.file))
        );
        var layer = comp.layers.add(footage);
        layer.name = entry.name + " [depth " + entry.depth + "]";
        layer.comment = "Parallax depth=" + entry.depth;
    }
    alert("Layers imported. Apply one shared camera/null and use the depth values in layer comments.");
    app.endUndoGroup();
}());
