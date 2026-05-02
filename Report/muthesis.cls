\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{muthesis}[2017/09/06 MU Thesis Class]

\RequirePackage{setspace}
\newif\ifwordctp@
\newif\ifsinglesp@
\newif\ifdoublesp@
\newif\ifanon@
\anon@false
\doublesp@false

\def\@thesis{dissertation}

\DeclareOption{anon}{\anon@true}
\DeclareOption{wordcount}{\wordctp@true}
\DeclareOption{singlespace}{\singlesp@true}
\DeclareOption{doublespace}{\doublesp@true}
\DeclareOption{PhD}{\def\degreetitle{Doctor of Philosophy} \def\@thesis{thesis}}
\DeclareOption{MSc}{\def\degreetitle{Master of Science} \def\@thesis{dissertation}}
\DeclareOption{MPhil}{\def\degreetitle{Master of Philosophy}}
\DeclareOption{MRes}{\def\degreetitle{Master of Research in Informatics}}
\DeclareOption{EngD}{\def\degreetitle{Doctor of Engineering} \def\@thesis{thesis}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}

\ExecuteOptions{PhD}
\ProcessOptions

\LoadClass[a4paper]{report}

\long\def\comment#1{}

\oddsidemargin 15truemm \evensidemargin 15truemm
\marginparwidth 40pt \marginparsep 10pt
\topmargin 0truemm \headsep 10truemm
\textheight 224truemm \textwidth 145truemm

\brokenpenalty=10000

\AtBeginDocument{%
\ifdoublesp@%
  \doublespacing%
\else\ifsinglesp@%
  \singlespacing%
\else%
  \onehalfspacing%
\fi%
\fi
}

\def\dept#1{\gdef\@dept{#1}}
\def\stuid#1{\gdef\@stuid{#1}}
\stuid{Need to set student id using stuid macro}
\def\department#1{\gdef\@dept{#1}}
\def\faculty#1{\gdef\@faculty{#1}}
\def\principaladviser#1{\gdef\@principaladviser{#1}}
\def\advis@r{Advisor}
\def\principaladvisor#1{\gdef\@principaladviser{#1}\gdef\advis@r{Advisor}}
\def\firstreader#1{\gdef\@firstreader{#1}}
\def\secondreader#1{\gdef\@secondreader{#1}}
\def\submitdate#1{\gdef\@submitdate{#1}}
\def\copyrightyear#1{\gdef\@copyrightyear{#1}}
\def\@title{}\def\@author{}
\def\@dept{Department of Computer Science}
\def\@faculty{Science and Engineering}
\def\@principaladviser{}\def\@firstreader{}\def\@secondreader{}
\def\@submitdate{\number\the\year}

\ifnum\month=12
    \@tempcnta=\year \advance\@tempcnta by 1
    \edef\@copyrightyear{\number\the\@tempcnta}
\else
    \def\@copyrightyear{\number\the\year}
\fi

\newif\ifcopyright \newif\iffigurespage \newif\iftablespage \newif\ifsigpage
\copyrighttrue \figurespagetrue \tablespagetrue \sigpagefalse

\def\titlep{%
  \thispagestyle{empty}%
  \null\vskip1in%
  \begin{center}
    \Huge\uppercase\expandafter{\@title}
  \end{center}
  \vfill
  \begin{center}
    \textsc{A \expandafter{\@thesis} submitted to The University of Manchester\\
      for the degree of Bachelors of Science\\
      in the Department of Computer Science}
  \end{center}
  \vfill
  \begin{center}
    \@submitdate
  \end{center}
  \vfill
  \begin{center}
    \normalfont
    \ifanon@{Student id: \@stuid}\else\@author\fi\\
  \end{center}
  \vfill
  \begin{center}
    \expandafter{\@dept}
  \end{center}\vskip.5in
  \newpage}

%% FIXED abstracttitle: store title in a box first to avoid
%% \expandafter interaction with \centerline and hyperref
\def\abstracttitle{%
  \par\bigskip
  \begingroup
  \singlespacing\large\centering
  \textsc{\@title}\par
  \@author\par
   \endgroup
   \bigskip\par
  }
\def\copyrightpage{%
\prefacesection{Copyright}
\begin{enumerate}
  \setlength{\itemsep}{-2pt}%
\item[i.]  The author of this thesis (including any appendices and/or
  schedules to this thesis) owns certain copyright or related rights
  in it (the ``Copyright'') and s/he has given The University of
  Manchester certain rights to use such Copyright, including for
  administrative purposes.
\item[ii.] Copies of this thesis, either in full or in extracts and
  whether in hard or electronic copy, may be made \textbf{only} in
  accordance with the Copyright, Designs and Patents Act 1988 (as
  amended) and regulations issued under it or, where appropriate, in
  accordance with licensing agreements which the University has from
  time to time. This page must form part of any such copies made.
\item[iii.] The ownership of certain Copyright, patents, designs,
  trade marks and other intellectual property (the ``Intellectual
  Property'') and any reproductions of copyright works in the thesis,
  for example graphs and tables (``Reproductions''), which may be
  described in this thesis, may not be owned by the author and may be
  owned by third parties. Such Intellectual Property and Reproductions
  cannot and must not be made available for use without the prior
  written permission of the owner(s) of the relevant Intellectual
  Property and/or Reproductions.
\item[iv.] Further information on the conditions under which
  disclosure, publication and commercialisation of this thesis, the
  Copyright and any Intellectual Property and/or Reproductions
  described in it may take place is available in the University IP
  Policy (see
  \url{http://documents.manchester.ac.uk/DocuInfo.aspx?DocID=24420}), in
    any relevant Thesis restriction declarations deposited in the
    University Library, The University Library's regulations (see
    \url{http://www.library.manchester.ac.uk/about/regulations/}) and
    in The University's policy on presentation of Theses
\end{enumerate}
\vfill\newpage}

\long\def\signature#1{%
\begin{center}
\begin{minipage}{4in}
\parindent=0pt
I certify that I have read this \expandafter{\@thesis} and that in my opinion
it is fully adequate, in scope and in quality, as a \expandafter{\@thesis} for the degree
of \degreetitle.
\par
\vspace{.5in}
\hbox to 4in{\hfil\shortstack{\vrule width 3in height 0.4pt\\#1}}
\end{minipage}
\end{center}}

\def\signaturepage{%
        \signature{\@principaladviser\\(Principal \advis@r)}
        \vfill
        \signature\@firstreader
        \vfill
        \signature\@secondreader
        \vfill
        \begin{center}
        \begin{minipage}{4in}
        Approved for the University Committee on Graduate Studies:\par
        \vspace{.5in}
        \hbox to 4in{\hfil\shortstack{\vrule width 3in height 0.4pt\\
                        Dean of Graduate Studies \& Research}}
        \end{minipage}
        \end{center}}

\def\declpage{%
\prefacesection{Declaration}
        \begin{center}
        \begin{minipage}{4in}
                \parindent=0pt
                No portion of the work referred to in this \expandafter{\@thesis} has been
                submitted in support of an application for another degree or
                qualification of this or any other university or other
                institute of learning.
                \par
        \end{minipage}%
        \end{center}%
        \vfill}

\def\beforeabstract{
        \pagenumbering{arabic}
        \pagestyle{plain}
        \titlep
        \ifsigpage\signaturepage\fi
        \tableofcontents
        \ifwordctp@%
        ~\\
        ~\\
        % TC:ignore
        \textbf{Word Count: \input{word.count}}
        % TC:endignore
        \fi
        \newpage
        \iftablespage
                \addvspace{10pt}
                \listoftables
                \newpage
        \fi
        \iffigurespage
                \addvspace{10pt}
                \listoffigures
                \newpage
        \fi}

\def\afterabstract{
        \newpage
        \declpage
        \ifcopyright\copyrightpage\fi
        }

\def\prefacesection#1{%
        \chapter*{#1}
        \addcontentsline{toc}{chapter}{#1}}

\def\afterpreface{\newpage
        \pagestyle{headings}}

\pagestyle{headings}

\def\thebibliography#1{\chapter*{\bibname\@mkboth
  {\uppercase{\bibname}}{\uppercase{\bibname}}}%
  \addcontentsline{toc}{chapter}{\bibname}%
  \list
  {\@biblabel{\arabic{enumiv}}}{\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \def\theenumiv{\arabic{enumiv}}}%
    \def\newblock{\hskip .11em plus.33em minus.07em}%
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=\@m}

\def\endthebibliography{%
  \def\@noitemerr{\@warning{Empty `thebibliography' environment}}%
  \endlist}