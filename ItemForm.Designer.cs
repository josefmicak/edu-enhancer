
namespace TAO_Enhancer
{
    partial class ItemForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.IdentifierLabel = new System.Windows.Forms.Label();
            this.TitleLabel = new System.Windows.Forms.Label();
            this.LabelLabel = new System.Windows.Forms.Label();
            this.QuestionLabel = new System.Windows.Forms.Label();
            this.CorrectAnswerLabel = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // IdentifierLabel
            // 
            this.IdentifierLabel.AutoSize = true;
            this.IdentifierLabel.Location = new System.Drawing.Point(29, 54);
            this.IdentifierLabel.Name = "IdentifierLabel";
            this.IdentifierLabel.Size = new System.Drawing.Size(82, 15);
            this.IdentifierLabel.TabIndex = 0;
            this.IdentifierLabel.Text = "IdentifierLabel";
            // 
            // TitleLabel
            // 
            this.TitleLabel.AutoSize = true;
            this.TitleLabel.Location = new System.Drawing.Point(29, 92);
            this.TitleLabel.Name = "TitleLabel";
            this.TitleLabel.Size = new System.Drawing.Size(57, 15);
            this.TitleLabel.TabIndex = 1;
            this.TitleLabel.Text = "TitleLabel";
            // 
            // LabelLabel
            // 
            this.LabelLabel.AutoSize = true;
            this.LabelLabel.Location = new System.Drawing.Point(29, 127);
            this.LabelLabel.Name = "LabelLabel";
            this.LabelLabel.Size = new System.Drawing.Size(63, 15);
            this.LabelLabel.TabIndex = 2;
            this.LabelLabel.Text = "LabelLabel";
            // 
            // QuestionLabel
            // 
            this.QuestionLabel.AutoSize = true;
            this.QuestionLabel.Location = new System.Drawing.Point(468, 54);
            this.QuestionLabel.Name = "QuestionLabel";
            this.QuestionLabel.Size = new System.Drawing.Size(83, 15);
            this.QuestionLabel.TabIndex = 3;
            this.QuestionLabel.Text = "QuestionLabel";
            // 
            // CorrectAnswerLabel
            // 
            this.CorrectAnswerLabel.AutoSize = true;
            this.CorrectAnswerLabel.Location = new System.Drawing.Point(469, 90);
            this.CorrectAnswerLabel.Name = "CorrectAnswerLabel";
            this.CorrectAnswerLabel.Size = new System.Drawing.Size(113, 15);
            this.CorrectAnswerLabel.TabIndex = 4;
            this.CorrectAnswerLabel.Text = "CorrectAnswerLabel";
            // 
            // ItemForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1223, 742);
            this.Controls.Add(this.CorrectAnswerLabel);
            this.Controls.Add(this.QuestionLabel);
            this.Controls.Add(this.LabelLabel);
            this.Controls.Add(this.TitleLabel);
            this.Controls.Add(this.IdentifierLabel);
            this.Name = "ItemForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "ItemForm";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label IdentifierLabel;
        private System.Windows.Forms.Label TitleLabel;
        private System.Windows.Forms.Label LabelLabel;
        private System.Windows.Forms.Label QuestionLabel;
        private System.Windows.Forms.Label CorrectAnswerLabel;
    }
}