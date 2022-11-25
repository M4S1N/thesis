from Logger import *

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

import numpy as np
import pywt

def reduceDICOM(data):
    new_data, _ = pywt.dwt2(data, 'haar', 'zpd')
    return new_data
    

def writeDICOM(data : np.ndarray, filename : str, path : str = "Images"):
    """
    Create a .DICOM file with the pixel values of the `data` array.

    # Params:
    
        `data` (np.ndarray): Image pixel array
        `filename` (str): Name that the .DICOM file will have
        `path` (str, optional): File path. Defaults to "Images".
    """
    # File meta info data elements
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 156
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'
    file_meta.MediaStorageSOPInstanceUID = '1.2.999.999.99.9.9999.9999.20030903150023'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
    file_meta.ImplementationClassUID = '1.2.888.888.88.8.8.8'

    # Main data elements
    ds = Dataset()
    ds.InstanceCreationDate = '20030903'
    ds.InstanceCreationTime = '150031'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'
    ds.SOPInstanceUID = '1.2.777.777.77.7.7777.7777.20030903150023'
    ds.StudyDate = '20030716'
    ds.StudyTime = '153557'
    ds.AccessionNumber = ''
    ds.Modality = 'RTPLAN'
    ds.Manufacturer = 'Manufacturer name here'
    ds.InstitutionName = 'Here'
    ds.ReferringPhysicianName = ''
    ds.StationName = 'COMPUTER002'
    ds.InstitutionalDepartmentName = 'Radiation Therap'
    ds.OperatorsName = 'operator'
    ds.ManufacturerModelName = 'Treatment Planning System name here'
    ds.PatientName = 'Last^First^mid^pre'
    ds.PatientID = 'id00001'
    ds.PatientBirthDate = ''
    ds.PatientSex = 'O'
    ds.SoftwareVersions = 'softwareV1'
    ds.StudyInstanceUID = '1.22.333.4.555555.6.7777777777777777777777777777'
    ds.SeriesInstanceUID = '1.2.333.444.55.6.7777.8888'
    ds.StudyID = 'study1'
    ds.SeriesNumber = '2'
    ds.RTPlanLabel = 'Plan1'
    ds.RTPlanName = 'Plan1'
    ds.RTPlanDate = '20030903'
    ds.RTPlanTime = '150023'
    ds.RTPlanGeometry = 'PATIENT'

    # Dose Reference Sequence
    dose_ref_sequence = Sequence()
    ds.DoseReferenceSequence = dose_ref_sequence

    # Dose Reference Sequence: Dose Reference 1
    dose_ref1 = Dataset()
    dose_ref1.DoseReferenceNumber = '1'
    dose_ref1.DoseReferenceStructureType = 'COORDINATES'
    dose_ref1.DoseReferenceDescription = 'iso'
    dose_ref1.DoseReferencePointCoordinates = [239.531250000000, 239.531250000000, -741.87000000000]
    dose_ref1.DoseReferenceType = 'ORGAN_AT_RISK'
    dose_ref1.DeliveryMaximumDose = '75.0'
    dose_ref1.OrganAtRiskMaximumDose = '75.0'
    dose_ref_sequence.append(dose_ref1)

    # Dose Reference Sequence: Dose Reference 2
    dose_ref2 = Dataset()
    dose_ref2.DoseReferenceNumber = '2'
    dose_ref2.DoseReferenceStructureType = 'COORDINATES'
    dose_ref2.DoseReferenceDescription = 'PTV'
    dose_ref2.DoseReferencePointCoordinates = [239.531250000000, 239.531250000000, -751.87000000000]
    dose_ref2.DoseReferenceType = 'TARGET'
    dose_ref2.TargetPrescriptionDose = '30.826203'
    dose_ref_sequence.append(dose_ref2)


    # Fraction Group Sequence
    frxn_gp_sequence = Sequence()
    ds.FractionGroupSequence = frxn_gp_sequence

    # Fraction Group Sequence: Fraction Group 1
    frxn_gp1 = Dataset()
    frxn_gp1.FractionGroupNumber = '1'
    frxn_gp1.NumberOfFractionsPlanned = '30'
    frxn_gp1.NumberOfBeams = '1'
    frxn_gp1.NumberOfBrachyApplicationSetups = '0'

    # Referenced Beam Sequence
    refd_beam_sequence = Sequence()
    frxn_gp1.ReferencedBeamSequence = refd_beam_sequence

    # Referenced Beam Sequence: Referenced Beam 1
    refd_beam1 = Dataset()
    refd_beam1.BeamDoseSpecificationPoint = [239.531250000000, 239.531250000000, -751.87000000000]
    refd_beam1.BeamDose = '1.0275401'
    refd_beam1.BeamMeterset = '116.0036697'
    refd_beam1.ReferencedBeamNumber = '1'
    refd_beam_sequence.append(refd_beam1)
    frxn_gp_sequence.append(frxn_gp1)


    # Beam Sequence
    beam_sequence = Sequence()
    ds.BeamSequence = beam_sequence

    # Beam Sequence: Beam 1
    beam1 = Dataset()
    beam1.Manufacturer = 'Linac co.'
    beam1.InstitutionName = 'Here'
    beam1.InstitutionalDepartmentName = 'Radiation Therap'
    beam1.ManufacturerModelName = 'Zapper9000'
    beam1.DeviceSerialNumber = '9999'
    beam1.TreatmentMachineName = 'unit001'
    beam1.PrimaryDosimeterUnit = 'MU'
    beam1.SourceAxisDistance = '1000.0'

    # Beam Limiting Device Sequence
    beam_limiting_device_sequence = Sequence()
    beam1.BeamLimitingDeviceSequence = beam_limiting_device_sequence

    # Beam Limiting Device Sequence: Beam Limiting Device 1
    beam_limiting_device1 = Dataset()
    beam_limiting_device1.RTBeamLimitingDeviceType = 'X'
    beam_limiting_device1.NumberOfLeafJawPairs = '1'
    beam_limiting_device_sequence.append(beam_limiting_device1)

    # Beam Limiting Device Sequence: Beam Limiting Device 2
    beam_limiting_device2 = Dataset()
    beam_limiting_device2.RTBeamLimitingDeviceType = 'Y'
    beam_limiting_device2.NumberOfLeafJawPairs = '1'
    beam_limiting_device_sequence.append(beam_limiting_device2)

    beam1.BeamNumber = '1'
    beam1.BeamName = 'Field 1'
    beam1.BeamType = 'STATIC'
    beam1.RadiationType = 'PHOTON'
    beam1.TreatmentDeliveryType = 'TREATMENT'
    beam1.NumberOfWedges = '0'
    beam1.NumberOfCompensators = '0'
    beam1.NumberOfBoli = '0'
    beam1.NumberOfBlocks = '0'
    beam1.FinalCumulativeMetersetWeight = '1.0'
    beam1.NumberOfControlPoints = '2'

    # Control Point Sequence
    cp_sequence = Sequence()
    beam1.ControlPointSequence = cp_sequence

    # Control Point Sequence: Control Point 0
    cp0 = Dataset()
    cp0.ControlPointIndex = '0'
    cp0.NominalBeamEnergy = '6.0'
    cp0.DoseRateSet = '650.0'

    # Beam Limiting Device Position Sequence
    beam_limiting_device_position_sequence = Sequence()
    cp0.BeamLimitingDevicePositionSequence = beam_limiting_device_position_sequence

    # Beam Limiting Device Position Sequence: Beam Limiting Device Position 1
    beam_limiting_device_position1 = Dataset()
    beam_limiting_device_position1.RTBeamLimitingDeviceType = 'X'
    beam_limiting_device_position1.LeafJawPositions = [-100.00000000000, 100.000000000000]
    beam_limiting_device_position_sequence.append(beam_limiting_device_position1)

    # Beam Limiting Device Position Sequence: Beam Limiting Device Position 2
    beam_limiting_device_position2 = Dataset()
    beam_limiting_device_position2.RTBeamLimitingDeviceType = 'Y'
    beam_limiting_device_position2.LeafJawPositions = [-100.00000000000, 100.000000000000]
    beam_limiting_device_position_sequence.append(beam_limiting_device_position2)

    cp0.GantryAngle = '0.0'
    cp0.GantryRotationDirection = 'NONE'
    cp0.BeamLimitingDeviceAngle = '0.0'
    cp0.BeamLimitingDeviceRotationDirection = 'NONE'
    cp0.PatientSupportAngle = '0.0'
    cp0.PatientSupportRotationDirection = 'NONE'
    cp0.TableTopEccentricAngle = '0.0'
    cp0.TableTopEccentricRotationDirection = 'NONE'
    cp0.TableTopVerticalPosition = None
    cp0.TableTopLongitudinalPosition = None
    cp0.TableTopLateralPosition = None
    cp0.IsocenterPosition = [235.711172833292, 244.135437110782, -724.97815409918]
    cp0.SourceToSurfaceDistance = '898.429664831309'
    cp0.CumulativeMetersetWeight = '0.0'

    # Referenced Dose Reference Sequence
    refd_dose_ref_sequence = Sequence()
    cp0.ReferencedDoseReferenceSequence = refd_dose_ref_sequence

    # Referenced Dose Reference Sequence: Referenced Dose Reference 1
    refd_dose_ref1 = Dataset()
    refd_dose_ref1.CumulativeDoseReferenceCoefficient = '0.0'
    refd_dose_ref1.ReferencedDoseReferenceNumber = '1'
    refd_dose_ref_sequence.append(refd_dose_ref1)

    # Referenced Dose Reference Sequence: Referenced Dose Reference 2
    refd_dose_ref2 = Dataset()
    refd_dose_ref2.CumulativeDoseReferenceCoefficient = '0.0'
    refd_dose_ref2.ReferencedDoseReferenceNumber = '2'
    refd_dose_ref_sequence.append(refd_dose_ref2)
    cp_sequence.append(cp0)

    # Control Point Sequence: Control Point 1
    cp1 = Dataset()
    cp1.ControlPointIndex = '1'
    cp1.CumulativeMetersetWeight = '1.0'

    # Referenced Dose Reference Sequence
    refd_dose_ref_sequence = Sequence()
    cp1.ReferencedDoseReferenceSequence = refd_dose_ref_sequence

    # Referenced Dose Reference Sequence: Referenced Dose Reference 1
    refd_dose_ref1 = Dataset()
    refd_dose_ref1.CumulativeDoseReferenceCoefficient = '0.9990268'
    refd_dose_ref1.ReferencedDoseReferenceNumber = '1'
    refd_dose_ref_sequence.append(refd_dose_ref1)

    # Referenced Dose Reference Sequence: Referenced Dose Reference 2
    refd_dose_ref2 = Dataset()
    refd_dose_ref2.CumulativeDoseReferenceCoefficient = '1.0'
    refd_dose_ref2.ReferencedDoseReferenceNumber = '2'
    refd_dose_ref_sequence.append(refd_dose_ref2)
    cp_sequence.append(cp1)

    beam1.ReferencedPatientSetupNumber = '1'
    beam_sequence.append(beam1)


    # Patient Setup Sequence
    patient_setup_sequence = Sequence()
    ds.PatientSetupSequence = patient_setup_sequence

    # Patient Setup Sequence: Patient Setup 1
    patient_setup1 = Dataset()
    patient_setup1.PatientPosition = 'HFS'
    patient_setup1.PatientSetupNumber = '1'
    patient_setup1.SetupTechniqueDescription = ''
    patient_setup_sequence.append(patient_setup1)


    # Referenced RT Plan Sequence
    refd_rt_plan_sequence = Sequence()
    ds.ReferencedRTPlanSequence = refd_rt_plan_sequence

    # Referenced RT Plan Sequence: Referenced RT Plan 1
    refd_rt_plan1 = Dataset()
    refd_rt_plan1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'
    refd_rt_plan1.ReferencedSOPInstanceUID = '1.9.999.999.99.9.9999.9999.20030903145128'
    refd_rt_plan1.RTPlanRelationship = 'PREDECESSOR'
    refd_rt_plan_sequence.append(refd_rt_plan1)


    # Referenced Structure Set Sequence
    refd_structure_set_sequence = Sequence()
    ds.ReferencedStructureSetSequence = refd_structure_set_sequence

    # Referenced Structure Set Sequence: Referenced Structure Set 1
    refd_structure_set1 = Dataset()
    refd_structure_set1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    refd_structure_set1.ReferencedSOPInstanceUID = '1.2.333.444.55.6.7777.88888'
    refd_structure_set_sequence.append(refd_structure_set1)

    ds.ApprovalStatus = 'UNAPPROVED'

    ds.file_meta = file_meta
    ds.is_implicit_VR = True
    ds.is_little_endian = True
    
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15

    ds.ImagesInAcquisition = "1"

    ds.Rows = data.shape[0]
    ds.Columns = data.shape[1]
    ds.InstanceNumber = ''

    # ds.ImagePositionPatient = r"0\0\1"
    ds.ImageOrientationPatient = ''
    ds.ImageType = ['DERIVED', 'SECONDARY']

    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    # ds.PixelSpacing = r"1\1"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1

    data = data * (1<<16)
    data = np.array(data, np.uint16)
    ds.PixelData = data.tobytes()
    
    ds.save_as(path+r'/'+filename, write_like_original=False)
    
    return ds