import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def assemble_bc(Model, LOADS, RF, PC):
    """

     	[GlobalStifMatrix, BB]  = assemble_bc(Model,LOADS,RF,PC)

     	Assembling procedure for SSFEM

     	Two steps :
         1. assembles MeanStifMatrix and the weighted StifMatrices{i}
			  for this purpose, loop over the elements
				. compute the localisation table, i.e
      		      the global number of dof for each local number of dof
               . compute the element mean- and weighted stiffness
                 matrices
               . assembles them.

         2. . from the Kmean and Ki, compute Kjk by weighted sums using
              cijk
            . Apply boundary conditions on Kjk --> Ljk
            . Assembles in GlobalStifMatrix
    """

    # ------------------------------------------------------------%%
    #       Extracting data from the data structures and initializing
    #       GlobalStifMatrix
    # ------------------------------------------------------------%%
    DdlPerNode = Model.DdlPerNode
    NbElts = Model.NbElts
    NodePerElt = Model.NodePerElt
    Nddl = Model.Nddl
    TypeDef = Model.TypeDef
    COORD = Model.COORD
    CONEC = Model.CONEC
    MATS = Model.MATS
    ELMAT = Model.ELMAT
    IntScheme = Model.IntSchemes[0]
    PsiSqNorm = PC.PC.PsiSqNorm
    PsiBasisSize = PC.PC.Size
    OrderGaussExp = RF.OrderExp
    # ------------------------------------------------------------
    #      Specific part switched according to random field nature
    # ------------------------------------------------------------
    if RF.Type == "Gaussian":
        ZZ = PC.PC.CC               # cijk coefficients used when
                                    # input is gaussian
        OrderExp = OrderGaussExp    # number of Ki's
    else:                           # dijk coefficient used when
        ZZ = PC.PC.DD               # input is expanded over PC
        OrderExp = PsiBasisSize     # number of Ki's

    GlobalStifMatrixSize = Nddl * PsiBasisSize
    GlobalStifMatrix = csr_matrix((int(GlobalStifMatrixSize), int(GlobalStifMatrixSize)))

    # ------------------------------------------------------------%%
    #     Assembling the mean stiffness matrix K0
    #     and the weighted stiffness matrices  Ki
    # ------------------------------------------------------------%%
    print("* Assembling the mean and weighted stiffness matrix :")
    MeanStifMatrix = csr_matrix((Nddl, Nddl))
    Force = np.zeros(Nddl)
    StifMatrices = []
    for k in range(1, OrderExp+1):
        StifMatrices.append(csr_matrix((int(Nddl), int(Nddl))))

    for elnum in range(1, NbElts+1):
        if elnum % int(NbElts / 10) == 0:
            print("Element: ", elnum)

        # Compute the localization table giving the global ddl number as a
        # function of the local ddl number in the element
        LOCE = np.zeros(NodePerElt * DdlPerNode)
        for i in range(0, NodePerElt):
            LOCE[2 * i] = 2 * CONEC[elnum, i]
            LOCE[2 * i + 1] = 2 * CONEC[elnum, i] + 1

        xy = np.zeros((NodePerElt, COORD.shape[1]))
        # Compute the element stiffness matrix
        for i in range(0, NodePerElt):
            xy[i, :] = COORD[CONEC[elnum, i], :]

        if ELMAT.type[elnum] == 1:
            ke_mean, ke, fe = stoch_quad4






