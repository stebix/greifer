"""Greifer Harden Listener for 3D Slicer.

Listens for OpenIGTLink StringMessages with device name "GreiferHarden"
and hardens the named transform into all volumes that reference it.

Usage:
    Slicer.exe --python-script greifer_harden_listener.py

Or add to ~/.slicerrc.py:
    exec(open("/path/to/greifer_harden_listener.py").read())
"""
import slicer
import vtk


@vtk.calldata_type(vtk.VTK_OBJECT)
def _on_node_added(caller, event, calldata):
    node = calldata
    if node is None or not node.IsA("vtkMRMLTextNode"):
        return
    if node.GetName() != "GreiferHarden":
        return

    transform_name = node.GetText()
    if not transform_name:
        return

    transform_node = slicer.util.getFirstNodeByName(
        transform_name, className="vtkMRMLLinearTransformNode"
    )
    if transform_node is None:
        print(f"[greifer] Transform node '{transform_name}' not found")
        slicer.mrmlScene.RemoveNode(node)
        return

    # Harden all volumes under this transform
    hardened = 0
    for i in range(slicer.mrmlScene.GetNumberOfNodes()):
        n = slicer.mrmlScene.GetNthNode(i)
        if n and n.IsA("vtkMRMLVolumeNode"):
            if n.GetParentTransformNode() == transform_node:
                logic = slicer.vtkSlicerTransformLogic()
                logic.hardenTransform(n)
                hardened += 1

    if hardened:
        print(f"[greifer] Hardened '{transform_name}' into {hardened} volume(s)")
    else:
        print(f"[greifer] No volumes found under '{transform_name}'")

    slicer.mrmlScene.RemoveNode(node)


def setup_harden_listener():
    tag = slicer.mrmlScene.AddObserver(
        slicer.vtkMRMLScene.NodeAddedEvent, _on_node_added
    )
    print(f"[greifer] Harden listener active (observer tag: {tag})")
    return tag


_greifer_harden_tag = setup_harden_listener()
